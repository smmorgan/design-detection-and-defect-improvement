"""
Data Loading Utilities for Design Mining
=========================================

Utilities for loading the datasets mentioned in the dissertation:
1. Stack Overflow Design Mining Dataset (Mahadi et al.)
2. TAWOS Dataset (JIRA tickets from open source projects)

Dataset locations from dissertation Table 1:
- Stack Overflow: https://zenodo.org/records/4010209
- TAWOS: https://github.com/SOLAR-group/TAWOS
"""

import os
import json
import logging
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Stack Overflow Dataset (Mahadi et al.)
# =============================================================================

class StackOverflowDataLoader:
    """Loader for Stack Overflow Design Mining Dataset.
    
    Source: https://zenodo.org/records/4010209
    
    This dataset contains 260,000 labelled discussions from Stack Overflow
    with 'design' and 'general' labels as described in Mahadi, Ernst, & Tongay (2021).
    """
    
    ZENODO_URL = "https://zenodo.org/records/4010209/files"
    
    def __init__(self, data_dir: str = "./data/stackoverflow"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, filename: str = "data.zip") -> Path:
        """Download the dataset from Zenodo."""
        output_path = self.data_dir / filename
        
        if output_path.exists():
            logger.info(f"Dataset already exists at {output_path}")
            return output_path
        
        url = f"{self.ZENODO_URL}/{filename}"
        logger.info(f"Downloading Stack Overflow dataset from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded to {output_path}")
        return output_path
    
    def extract(self, archive_path: Path) -> Path:
        """Extract the downloaded archive."""
        extract_dir = self.data_dir / "extracted"
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        
        logger.info(f"Extracted to {extract_dir}")
        return extract_dir
    
    def load(self, filepath: Optional[str] = None) -> Tuple[List[str], List[int]]:
        """Load the Stack Overflow dataset.
        
        Returns:
            texts: List of text content
            labels: List of labels (1 = design, 0 = general)
        """
        if filepath:
            data_path = Path(filepath)
        else:
            # Look for common file patterns
            for pattern in ['*.csv', '*.json', '*.jsonl']:
                files = list(self.data_dir.glob(f"**/{pattern}"))
                if files:
                    data_path = files[0]
                    break
            else:
                raise FileNotFoundError(f"No data files found in {self.data_dir}")
        
        logger.info(f"Loading data from {data_path}")
        
        texts = []
        labels = []
        
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            # Adjust column names based on actual dataset structure
            text_col = next((c for c in df.columns if 'text' in c.lower() or 'body' in c.lower()), df.columns[0])
            label_col = next((c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()), df.columns[-1])
            
            texts = df[text_col].tolist()
            # Convert labels to binary
            labels = df[label_col].apply(lambda x: 1 if str(x).lower() == 'design' else 0).tolist()
            
        elif data_path.suffix in ['.json', '.jsonl']:
            with open(data_path, 'r') as f:
                if data_path.suffix == '.jsonl':
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            for item in data:
                texts.append(item.get('text', item.get('body', '')))
                label = item.get('label', item.get('class', 0))
                labels.append(1 if str(label).lower() == 'design' else 0)
        
        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Design samples: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        
        return texts, labels
    
    def load_from_local(self, csv_path: str) -> Tuple[List[str], List[int]]:
        """Load from a local CSV file with expected format."""
        df = pd.read_csv(csv_path)
        
        # Expected columns based on Mahadi et al. dataset structure
        required_cols = ['text', 'label']
        
        # Map actual column names
        col_mapping = {}
        for req_col in required_cols:
            for actual_col in df.columns:
                if req_col.lower() in actual_col.lower():
                    col_mapping[req_col] = actual_col
                    break
        
        texts = df[col_mapping.get('text', df.columns[0])].fillna('').tolist()
        labels_raw = df[col_mapping.get('label', df.columns[-1])].tolist()
        
        # Convert to binary
        labels = [1 if str(l).lower() in ['design', '1', 'true'] else 0 for l in labels_raw]
        
        return texts, labels


# =============================================================================
# TAWOS Dataset
# =============================================================================

class TAWOSDataLoader:
    """Loader for TAWOS Dataset (JIRA tickets from Agile Open Source Projects).
    
    Source: https://github.com/SOLAR-group/TAWOS
    
    Contains 458,232 issues from 39 open-source projects including:
    - Issue title
    - Issue description
    - Comments
    - Effort information
    - Relationships to other issues
    """
    
    GITHUB_URL = "https://github.com/SOLAR-group/TAWOS"
    
    def __init__(self, data_dir: str = "./data/tawos"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_project(self, project_dir: Path) -> pd.DataFrame:
        """Load a single project's data."""
        issues = []
        
        # Look for JSON files in project directory
        for json_file in project_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        issues.extend(data)
                    else:
                        issues.append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse {json_file}")
        
        return pd.DataFrame(issues)
    
    def load_all_projects(self) -> pd.DataFrame:
        """Load all projects in the TAWOS dataset."""
        all_dfs = []
        
        for project_dir in self.data_dir.iterdir():
            if project_dir.is_dir():
                logger.info(f"Loading project: {project_dir.name}")
                df = self.load_project(project_dir)
                if len(df) > 0:
                    df['project'] = project_dir.name
                    all_dfs.append(df)
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"Loaded {len(combined)} total issues from {len(all_dfs)} projects")
            return combined
        else:
            return pd.DataFrame()
    
    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load TAWOS data from a preprocessed CSV file."""
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} issues from {csv_path}")
        return df
    
    def preprocess_for_classification(
        self, 
        df: pd.DataFrame,
        title_col: str = 'summary',
        description_col: str = 'description',
        comments_col: str = 'comments'
    ) -> Tuple[List[str], List[str]]:
        """Preprocess TAWOS data for design mining classification.
        
        Combines title, description, and comments as per dissertation methodology.
        
        Returns:
            texts: Combined text for each issue
            issue_keys: Issue identifiers for tracking
        """
        texts = []
        issue_keys = []
        
        for idx, row in df.iterrows():
            parts = []
            
            # Add title/summary
            if title_col in df.columns and pd.notna(row.get(title_col)):
                parts.append(str(row[title_col]))
            
            # Add description
            if description_col in df.columns and pd.notna(row.get(description_col)):
                parts.append(str(row[description_col]))
            
            # Add comments (may be a list or string)
            if comments_col in df.columns and pd.notna(row.get(comments_col)):
                comments = row[comments_col]
                if isinstance(comments, list):
                    parts.extend([str(c) for c in comments if c])
                else:
                    parts.append(str(comments))
            
            # Combine with [SEP] token
            text = ' [SEP] '.join(parts)
            texts.append(text)
            
            # Get issue key/ID
            key = row.get('key', row.get('id', str(idx)))
            issue_keys.append(str(key))
        
        logger.info(f"Preprocessed {len(texts)} issues for classification")
        return texts, issue_keys
    
    def get_issue_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract metadata useful for analysis (timestamps, issue types, etc.)."""
        metadata_cols = [
            'key', 'id', 'project', 'issuetype', 'priority', 
            'status', 'resolution', 'created', 'updated', 'resolved',
            'storypoints', 'timeoriginalestimate', 'timespent'
        ]
        
        available_cols = [c for c in metadata_cols if c in df.columns]
        return df[available_cols].copy()


# =============================================================================
# Combined Data Loader
# =============================================================================

class DesignMiningDataLoader:
    """Combined loader for all design mining datasets."""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.so_loader = StackOverflowDataLoader(self.base_dir / "stackoverflow")
        self.tawos_loader = TAWOSDataLoader(self.base_dir / "tawos")
    
    def load_stackoverflow(self, filepath: Optional[str] = None) -> Tuple[List[str], List[int]]:
        """Load Stack Overflow dataset."""
        return self.so_loader.load(filepath)
    
    def load_tawos(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load TAWOS dataset."""
        if filepath:
            return self.tawos_loader.load_from_csv(filepath)
        return self.tawos_loader.load_all_projects()
    
    def prepare_transfer_learning_data(
        self,
        so_path: str,
        tawos_path: str
    ) -> Dict:
        """Prepare data for transfer learning pipeline.
        
        Returns dict with:
        - stackoverflow_texts, stackoverflow_labels
        - tawos_texts, tawos_keys
        """
        # Load Stack Overflow
        so_texts, so_labels = self.load_stackoverflow(so_path)
        
        # Load and preprocess TAWOS
        tawos_df = self.load_tawos(tawos_path)
        tawos_texts, tawos_keys = self.tawos_loader.preprocess_for_classification(tawos_df)
        
        return {
            'stackoverflow_texts': so_texts,
            'stackoverflow_labels': so_labels,
            'tawos_texts': tawos_texts,
            'tawos_keys': tawos_keys,
            'tawos_metadata': self.tawos_loader.get_issue_metadata(tawos_df)
        }


# =============================================================================
# Sample Data for Testing
# =============================================================================

def create_sample_stackoverflow_csv(output_path: str, n_samples: int = 1000):
    """Create a sample CSV file in Stack Overflow format for testing."""
    
    design_templates = [
        "What's the best architecture pattern for building a scalable microservices system?",
        "Should I use event-driven architecture or request-response for my backend?",
        "Design decision: how to structure the data layer in a clean architecture?",
        "What are the tradeoffs between monolith and microservices architecture?",
        "How should I design the API gateway for my distributed system?",
        "Best practices for designing a message queue system architecture",
        "System design: how to implement CQRS pattern effectively?",
        "Architecture review: evaluating hexagonal vs layered architecture",
        "Design pattern recommendation for handling complex business rules",
        "How to architect a real-time notification system at scale?"
    ]
    
    general_templates = [
        "How do I fix this null pointer exception in my Java code?",
        "Why is my React component not re-rendering?",
        "Bug: CSS styles not applying to nested elements",
        "Performance issue: slow database query taking 30 seconds",
        "How to parse JSON in Python?",
        "TypeError when calling async function in JavaScript",
        "Unit test failing intermittently, need help debugging",
        "How to install npm packages globally?",
        "Git merge conflict, how to resolve?",
        "Memory leak in my Node.js application"
    ]
    
    import random
    random.seed(42)
    
    data = []
    for _ in range(n_samples):
        is_design = random.random() < 0.3
        
        if is_design:
            text = random.choice(design_templates)
            label = 'design'
        else:
            text = random.choice(general_templates)
            label = 'general'
        
        # Add some variation
        text += f" [Project: {random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'])}]"
        
        data.append({'text': text, 'label': label})
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample Stack Overflow dataset at {output_path}")
    return output_path


def create_sample_tawos_csv(output_path: str, n_samples: int = 500):
    """Create a sample CSV file in TAWOS format for testing."""
    
    summaries = [
        "Implement user authentication module",
        "Fix bug in payment processing",
        "Design new dashboard layout",
        "Refactor database connection pool",
        "Add unit tests for API endpoints",
        "Architecture: migrate to microservices",
        "Update documentation for REST API",
        "Performance optimization for search",
        "Design decision: caching strategy",
        "Bug: login fails on mobile devices"
    ]
    
    descriptions = [
        "We need to implement a new feature that allows users to...",
        "The system is currently experiencing issues when...",
        "After reviewing the architecture, we propose to...",
        "This task involves updating the existing...",
        "Based on the requirements, we need to design..."
    ]
    
    import random
    random.seed(42)
    
    data = []
    for i in range(n_samples):
        data.append({
            'key': f'PROJ-{i+1}',
            'summary': random.choice(summaries),
            'description': random.choice(descriptions),
            'issuetype': random.choice(['Bug', 'Story', 'Task', 'Epic']),
            'priority': random.choice(['High', 'Medium', 'Low']),
            'status': random.choice(['Open', 'In Progress', 'Done']),
            'project': random.choice(['ProjectA', 'ProjectB', 'ProjectC']),
            'storypoints': random.choice([1, 2, 3, 5, 8, 13])
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample TAWOS dataset at {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data loading utilities')
    parser.add_argument('--create-samples', action='store_true', 
                       help='Create sample datasets for testing')
    parser.add_argument('--output-dir', type=str, default='./data/samples',
                       help='Output directory for sample data')
    
    args = parser.parse_args()
    
    if args.create_samples:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        create_sample_stackoverflow_csv(output_dir / 'stackoverflow_sample.csv')
        create_sample_tawos_csv(output_dir / 'tawos_sample.csv')
        
        print(f"\nSample datasets created in {output_dir}")
        print("Use these for testing the training pipeline.")
