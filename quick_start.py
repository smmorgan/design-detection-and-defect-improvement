#!/usr/bin/env python3
"""
Quick Start Script for Design Mining Training
==============================================

This script provides a simple entry point to:
1. Set up the environment
2. Create sample data (for testing)
3. Run a training demo

Usage:
    python quick_start.py [--full-demo] [--create-samples-only]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    required = ['torch', 'transformers', 'sklearn', 'pandas', 'numpy', 'tqdm']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True


def create_sample_data():
    """Create sample datasets for demo."""
    from data_loaders import create_sample_stackoverflow_csv, create_sample_tawos_csv
    
    data_dir = Path('./data/samples')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating sample datasets...")
    create_sample_stackoverflow_csv(data_dir / 'stackoverflow_sample.csv', n_samples=800)
    create_sample_tawos_csv(data_dir / 'tawos_sample.csv', n_samples=400)
    
    print(f"Sample data created in {data_dir}")
    return data_dir


def run_demo_training(epochs: int = 3):
    """Run demo training with sample data."""
    print("\n" + "="*60)
    print("Starting Demo Training")
    print("="*60)
    
    # Import here to avoid loading if dependencies missing
    from train_design_classifier import (
        Config, DesignMiningTrainer, BaselineComparison,
        MetricsCalculator, generate_sample_data
    )
    from sklearn.model_selection import train_test_split
    import torch
    
    # Configuration
    config = Config()
    config.NUM_EPOCHS = epochs
    config.BATCH_SIZE = 8  # Smaller for demo
    
    print(f"\nDevice: {config.DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    
    # Generate sample data
    print("\nGenerating sample data...")
    texts, labels = generate_sample_data(n_samples=300)
    
    # Split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Initialize trainer
    print("\nInitializing BERT model...")
    trainer = DesignMiningTrainer(config)
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    test_loss, test_metrics = trainer.evaluate(test_loader)
    MetricsCalculator.print_metrics(test_metrics, "BERT Test")
    
    # Baseline comparison
    print("\nTraining baseline models for comparison...")
    baseline_results = BaselineComparison.train_baseline_models(
        train_texts, train_labels,
        test_texts, test_labels
    )
    
    # Summary
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'AUC':<12}")
    print("-" * 60)
    print(f"{'BERT':<25} {test_metrics['accuracy']:<12.4f} "
          f"{test_metrics['f1_score']:<12.4f} {test_metrics.get('auc', 0):<12.4f}")
    for model_name, metrics in baseline_results.items():
        print(f"{model_name:<25} {metrics['accuracy']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {metrics.get('auc', 0):<12.4f}")
    
    # Save model
    output_dir = './model_output_demo'
    trainer.save_model(output_dir)
    print(f"\nModel saved to: {output_dir}")
    
    # Demo predictions
    print("\n" + "="*60)
    print("Sample Predictions")
    print("="*60)
    
    sample_texts = [
        "We need to design the architecture for the new microservices system using event-driven patterns",
        "Fix bug: null pointer exception when user clicks submit button",
        "Architecture decision: should we use CQRS pattern for the data layer?",
        "Update npm dependencies to latest versions"
    ]
    
    predictions, confidences = trainer.predict(sample_texts)
    
    for text, pred, conf in zip(sample_texts, predictions, confidences):
        label = "DESIGN" if pred == 1 else "GENERAL"
        print(f"\n[{label}] (confidence: {conf:.2f})")
        print(f"  {text[:70]}...")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Download real datasets (see README.md)")
    print("2. Run full training: python train_design_classifier.py --mode full")
    print("3. For transfer learning: python train_design_classifier.py --mode transfer")


def main():
    parser = argparse.ArgumentParser(
        description='Quick start for Design Mining training'
    )
    parser.add_argument(
        '--full-demo',
        action='store_true',
        help='Run full demo with more epochs (5 instead of 3)'
    )
    parser.add_argument(
        '--create-samples-only',
        action='store_true',
        help='Only create sample data files, do not train'
    )
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip dependency check'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Design Mining for Software Architecture Issues")
    print("Quick Start Script")
    print("="*60)
    
    # Check dependencies
    if not args.skip_check and not check_dependencies():
        print("\nInstalling required packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    if args.create_samples_only:
        create_sample_data()
        return
    
    # Run demo
    epochs = 5 if args.full_demo else 3
    run_demo_training(epochs=epochs)


if __name__ == "__main__":
    main()
