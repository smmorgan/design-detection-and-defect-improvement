"""
TAWOS Database Connector
========================

Connects to the TAWOS MySQL database to fetch JIRA issues for design mining.

TAWOS (Technical and Architectural Work Observed in Software) contains
JIRA issues from various Apache projects that can be used for transfer
learning in design classification.

Database schema reference: https://github.com/SOLAR-group/TAWOS

TAWOS Schema:
    Issue: ID, Issue_Key, Title, Description, Description_Text, Type,
           Priority, Status, Resolution, Creation_Date, Project_ID, ...
    Comment: ID, Comment, Comment_Text, Creation_Date, Issue_ID
    Project: ID, Project_Key, Name, URL, Description
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    MySQLError = Exception

logger = logging.getLogger(__name__)


@dataclass
class TAWOSConfig:
    """Configuration for TAWOS database connection."""
    host: str = "localhost"
    port: int = 3306
    database: str = "tawos"
    user: str = "root"
    password: str = ""

    # Query parameters
    min_text_length: int = 50
    max_issues: Optional[int] = None
    projects: List[str] = field(default_factory=list)
    include_issue_type: bool = False
    issue_types: Optional[List[str]] = None  # None = all types; set list to filter


class TAWOSConnector:
    """Connector for fetching JIRA issues from TAWOS MySQL database.

    TAWOS contains structured JIRA issue data including:
    - Issue key, type, status, priority
    - Title and description (raw and text-only variants)
    - Comments (raw and text-only variants)
    - Project information via Project table
    """

    def __init__(self, config: TAWOSConfig):
        """Initialize connector with configuration.

        Args:
            config: TAWOSConfig with database connection parameters
        """
        if not MYSQL_AVAILABLE:
            raise ImportError(
                "mysql-connector-python is required for TAWOS connectivity. "
                "Install with: pip install mysql-connector-python"
            )

        self.config = config
        self.connection = None

    def connect(self) -> bool:
        """Establish connection to TAWOS database.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                charset='utf8mb4',
                use_unicode=True
            )

            if self.connection.is_connected():
                db_info = self.connection.server_info
                logger.info(f"Connected to TAWOS MySQL Server version {db_info}")

                cursor = self.connection.cursor()
                cursor.execute("SELECT DATABASE();")
                db_name = cursor.fetchone()[0]
                logger.info(f"Connected to database: {db_name}")
                cursor.close()

                return True

        except MySQLError as e:
            logger.error(f"Error connecting to TAWOS database: {e}")
            return False

    def disconnect(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("TAWOS database connection closed")

    def get_schema_info(self) -> Dict[str, List[str]]:
        """Get table and column information from the database.

        Returns:
            Dictionary mapping table names to list of columns
        """
        if not self.connection or not self.connection.is_connected():
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()

        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            cursor.execute(f"DESCRIBE `{table}`")
            columns = [row[0] for row in cursor.fetchall()]
            schema[table] = columns

        cursor.close()
        return schema

    def fetch_issues(
        self,
        include_comments: bool = False,
        include_issue_type: Optional[bool] = None,
        projects: Optional[List[str]] = None,
        issue_types: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch JIRA issues from TAWOS database.

        Args:
            include_comments: Whether to fetch and concatenate comments
            projects: List of project keys to filter (e.g., ['HADOOP', 'SPARK'])
            issue_types: List of issue types to include
            limit: Maximum number of issues to fetch

        Returns:
            DataFrame with columns: issue_key, project, issue_type, summary,
            description, text (combined), and optionally comments
        """
        if not self.connection or not self.connection.is_connected():
            raise ConnectionError("Not connected to database")

        projects = projects or self.config.projects
        issue_types = issue_types or self.config.issue_types
        limit = limit or self.config.max_issues
        if include_issue_type is None:
            include_issue_type = self.config.include_issue_type

        cursor = self.connection.cursor(dictionary=True)

        # Query using actual TAWOS schema:
        #   Issue table: ID, Issue_Key, Title, Description_Text, Type, ...
        #   Project table: ID, Project_Key (joined via Issue.Project_ID)
        query = """
            SELECT
                i.ID as id,
                i.Issue_Key as issue_key,
                p.Project_Key as project,
                i.Type as issue_type,
                i.Title as summary,
                i.Description_Text as description,
                i.Status as status,
                i.Priority as priority,
                i.Creation_Date as created,
                i.Last_Updated as updated
            FROM Issue i
            JOIN Project p ON i.Project_ID = p.ID
            WHERE i.Title IS NOT NULL
            AND i.Description_Text IS NOT NULL
            AND CHAR_LENGTH(CONCAT(COALESCE(i.Title, ''), ' ', COALESCE(i.Description_Text, ''))) >= %s
        """
        params = [self.config.min_text_length]

        # Add project filter
        if projects:
            placeholders = ", ".join(["%s"] * len(projects))
            query += f" AND p.Project_Key IN ({placeholders})"
            params.extend(projects)

        # Add issue type filter
        if issue_types:
            placeholders = ", ".join(["%s"] * len(issue_types))
            query += f" AND i.Type IN ({placeholders})"
            params.extend(issue_types)

        query += " ORDER BY i.Creation_Date DESC"
        if limit:
            query += " LIMIT %s"
            params.append(limit)

        logger.info("Fetching issues from TAWOS...")
        cursor.execute(query, params)
        issues = cursor.fetchall()
        logger.info(f"Fetched {len(issues)} issues")

        df = pd.DataFrame(issues)

        if df.empty:
            logger.warning("No issues found matching criteria")
            cursor.close()
            return df

        # Combine title and description into text field
        if include_issue_type:
            df['text'] = (
                '[TYPE: ' + df['issue_type'].fillna('Unknown') + '] '
                + df['summary'].fillna('') + ' [SEP] ' + df['description'].fillna('')
            )
        else:
            df['text'] = df['summary'].fillna('') + ' [SEP] ' + df['description'].fillna('')

        # Fetch comments if requested
        if include_comments and len(df) > 0:
            df = self._add_comments(df, cursor)

        cursor.close()

        # Log distribution info
        if 'issue_type' in df.columns:
            type_dist = df['issue_type'].value_counts()
            logger.info("Issue type distribution:")
            for issue_type, count in type_dist.items():
                logger.info(f"  {issue_type}: {count}")

        return df

    def _add_comments(self, df: pd.DataFrame, cursor) -> pd.DataFrame:
        """Add comments to issues DataFrame.

        Uses Comment table joined by Issue_ID, fetching Comment_Text
        (the cleaned text version of comments).

        Args:
            df: DataFrame with issues (must have 'id' column)
            cursor: Database cursor

        Returns:
            DataFrame with comments column added
        """
        issue_ids = df['id'].tolist()

        # Batch fetch comments using Issue_ID foreign key
        placeholders = ", ".join(["%s"] * len(issue_ids))
        comment_query = f"""
            SELECT Issue_ID, Comment_Text
            FROM Comment
            WHERE Issue_ID IN ({placeholders})
            ORDER BY Creation_Date
        """

        cursor.execute(comment_query, issue_ids)
        comments = cursor.fetchall()

        # Group comments by issue ID
        comments_by_issue = {}
        for row in comments:
            issue_id = row['Issue_ID']
            body = row['Comment_Text'] or ''
            if body.strip():
                if issue_id not in comments_by_issue:
                    comments_by_issue[issue_id] = []
                comments_by_issue[issue_id].append(body)

        # Add comments to DataFrame
        df['comments'] = df['id'].apply(
            lambda i: ' [SEP] '.join(comments_by_issue.get(i, []))
        )

        # Update text field to include comments
        has_comments = df['comments'].str.len() > 0
        df.loc[has_comments, 'text'] = (
            df.loc[has_comments, 'text'] + ' [SEP] ' + df.loc[has_comments, 'comments']
        )

        comment_count = sum(1 for v in comments_by_issue.values() if v)
        logger.info(f"Added comments for {comment_count} issues "
                     f"({len(comments)} total comments)")

        return df

    def get_issue_count(self, projects: Optional[List[str]] = None) -> int:
        """Get total count of issues matching criteria.

        Args:
            projects: Optional list of project keys to filter

        Returns:
            Count of matching issues
        """
        if not self.connection or not self.connection.is_connected():
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()

        if projects:
            placeholders = ", ".join(["%s"] * len(projects))
            query = f"""
                SELECT COUNT(*)
                FROM Issue i
                JOIN Project p ON i.Project_ID = p.ID
                WHERE i.Title IS NOT NULL AND i.Description_Text IS NOT NULL
                AND p.Project_Key IN ({placeholders})
            """
            cursor.execute(query, projects)
        else:
            query = """
                SELECT COUNT(*)
                FROM Issue
                WHERE Title IS NOT NULL AND Description_Text IS NOT NULL
            """
            cursor.execute(query)

        count = cursor.fetchone()[0]
        cursor.close()

        return count

    def list_projects(self) -> List[Dict]:
        """List all available projects in the database.

        Returns:
            List of dicts with project info (project_key, name, issue_count)
        """
        if not self.connection or not self.connection.is_connected():
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor(dictionary=True)

        query = """
            SELECT
                p.Project_Key as project_key,
                p.Name as name,
                COUNT(i.ID) as issue_count
            FROM Project p
            LEFT JOIN Issue i ON i.Project_ID = p.ID
            GROUP BY p.ID, p.Project_Key, p.Name
            ORDER BY issue_count DESC
        """

        cursor.execute(query)
        projects = cursor.fetchall()
        cursor.close()

        return projects


def fetch_tawos_data(
    host: str = "localhost",
    port: int = 3306,
    database: str = "tawos",
    user: str = "root",
    password: str = "",
    projects: Optional[List[str]] = None,
    include_comments: bool = False,
    max_issues: Optional[int] = None
) -> Tuple[List[str], pd.DataFrame]:
    """Convenience function to fetch TAWOS data for transfer learning.

    Args:
        host: MySQL host
        port: MySQL port
        database: Database name
        user: Database user
        password: Database password
        projects: Optional list of project keys to filter
        include_comments: Whether to include comment text
        max_issues: Maximum number of issues to fetch

    Returns:
        Tuple of (texts list, full DataFrame)
    """
    config = TAWOSConfig(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        projects=projects or [],
        max_issues=max_issues
    )

    connector = TAWOSConnector(config)

    if not connector.connect():
        raise ConnectionError("Failed to connect to TAWOS database")

    try:
        df = connector.fetch_issues(include_comments=include_comments)
        texts = df['text'].tolist() if not df.empty else []
        return texts, df
    finally:
        connector.disconnect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test TAWOS database connection")
    parser.add_argument("--host", default=os.environ.get('TAWOS_DB_HOST', 'localhost'),
                        help="MySQL host (env: TAWOS_DB_HOST)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get('TAWOS_DB_PORT', '3306')),
                        help="MySQL port (env: TAWOS_DB_PORT)")
    parser.add_argument("--database", default=os.environ.get('TAWOS_DB_NAME', 'tawos'),
                        help="Database name (env: TAWOS_DB_NAME)")
    parser.add_argument("--user", default=os.environ.get('TAWOS_DB_USER', 'root'),
                        help="Database user (env: TAWOS_DB_USER)")
    parser.add_argument("--password", default=os.environ.get('TAWOS_DB_PASSWORD', ''),
                        help="Database password (env: TAWOS_DB_PASSWORD)")
    parser.add_argument("--list-projects", action="store_true",
                        help="List available projects")
    parser.add_argument("--schema", action="store_true",
                        help="Show database schema")
    parser.add_argument("--sample", type=int, default=5,
                        help="Fetch sample issues")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = TAWOSConfig(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password
    )

    connector = TAWOSConnector(config)

    if connector.connect():
        try:
            if args.schema:
                print("\n=== Database Schema ===")
                schema = connector.get_schema_info()
                for table, columns in schema.items():
                    print(f"\n{table}:")
                    for col in columns:
                        print(f"  - {col}")

            if args.list_projects:
                print("\n=== Available Projects ===")
                projects = connector.list_projects()
                for p in projects:
                    print(f"  {p['project_key']} ({p['name']}): {p['issue_count']} issues")

            if args.sample > 0:
                print(f"\n=== Sample Issues (first {args.sample}) ===")
                df = connector.fetch_issues(limit=args.sample)
                for _, row in df.iterrows():
                    print(f"\n[{row['issue_key']}] {row['issue_type']}")
                    print(f"  Title: {row['summary'][:100]}...")
                    desc = str(row.get('description', ''))
                    if desc:
                        print(f"  Description: {desc[:150]}...")
        finally:
            connector.disconnect()
    else:
        print("Failed to connect to database")
