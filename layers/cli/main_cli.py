"""Command-line interface for Myanmar LNP Dataset."""

import click
from rich import print as rprint
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version(version="0.1.0")
def main():
    """Myanmar LNP Dataset CLI.
    
    Tools for working with Myanmar language NLP datasets.
    """
    pass


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--format", type=click.Choice(["jsonl", "csv", "parquet"]), default="jsonl")
def preprocess(input_file, output_file, format):
    """Preprocess Myanmar text data."""
    from api.data_loader import load_data
    from api.preprocess import MyanmarPreprocessor
    
    rprint(f"[bold blue]Loading data from {input_file}...[/bold blue]")
    df = load_data(input_file)
    
    rprint(f"[bold green]Preprocessing {len(df)} records...[/bold green]")
    
    preprocessor = MyanmarPreprocessor()
    
    if "text" in df.columns:
        df["text"] = preprocessor.transform(df["text"].tolist())
    
    from api.data_loader import save_data
    save_data(df, output_file, format)
    
    rprint(f"[bold green]Saved to {output_file}[/bold green]")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--prob", type=float, default=0.15, help="Augmentation probability")
def augment(input_file, output_file, prob):
    """Augment Myanmar text data."""
    from api.data_loader import load_data, save_data
    from api.augment import MyanmarAugmenter
    
    rprint(f"[bold blue]Loading data from {input_file}...[/bold blue]")
    df = load_data(input_file)
    
    rprint(f"[bold green]Augmenting {len(df)} records...[/bold green]")
    
    augmenter = MyanmarAugmenter(synonym_prob=prob)
    
    if "text" in df.columns:
        df["text"] = augmenter.augment_batch(df["text"].tolist())
    
    save_data(df, output_file)
    
    rprint(f"[bold green]Saved to {output_file}[/bold green]")


@main.command()
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--model", type=str, default="logistic_regression")
@click.option("--epochs", type=int, default=10)
def train(train_file, test_file, model, epochs):
    """Train a classification model."""
    from api.data_loader import load_data
    from api.vectorizer import create_vectorizer
    from api.models.classifier import MyanmarTextClassifier
    from api.models.trainer import ModelTrainer, create_dataloader
    
    console.print(f"[bold blue]Loading training data...[/bold blue]")
    train_df = load_data(train_file)
    test_df = load_data(test_file)
    
    console.print(f"[bold blue]Vectorizing...[/bold blue]")
    vectorizer = create_vectorizer("tfidf", max_features=10000)
    
    X_train = vectorizer.fit_transform(train_df["text"].tolist())
    X_test = vectorizer.transform(test_df["text"].tolist())
    
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    
    console.print(f"[bold green]Training {model} for {epochs} epochs...[/bold green]")
    
    # Create PyTorch model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    import numpy as np
    from api.models.classifier import MyanmarTextClassifier
    
    pytorch_model = MyanmarTextClassifier(input_dim, num_classes)
    trainer = ModelTrainer(pytorch_model, device="cpu")
    
    train_loader = create_dataloader(X_train, y_train, batch_size=32)
    val_loader = create_dataloader(X_test, y_test, batch_size=32, shuffle=False)
    
    history = trainer.train(train_loader, val_loader, num_epochs=epochs)
    
    console.print("[bold green]Training complete![/bold green]")


@main.command()
@click.argument("data_file", type=click.Path(exists=True))
def stats(data_file):
    """Show dataset statistics."""
    from api.data_loader import load_data
    import numpy as np
    
    df = load_data(data_file)
    
    table = Table(title="Dataset Statistics")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Records", str(len(df)))
    
    if "label" in df.columns:
        label_counts = df["label"].value_counts()
        table.add_row("Unique Labels", str(len(label_counts)))
    
    if "text" in df.columns:
        avg_length = df["text"].str.len().mean()
        table.add_row("Avg Text Length", f"{avg_length:.1f}")
    
    console.print(table)


@main.command()
@click.argument("model_file", type=click.Path(exists=True))
@click.argument("input_text", type=str))
def predict(model_file, input_text):
    """Make prediction on input text."""
    rprint(f"[bold blue]Predicting: {input_text}[/bold blue]")
    rprint("[yellow]Model loading not yet implemented[/yellow]")


if __name__ == "__main__":
    main()