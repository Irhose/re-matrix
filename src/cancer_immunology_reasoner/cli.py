from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from loguru import logger

from cancer_immunology_reasoner.ingestion import ingest_corpus
from cancer_immunology_reasoner.pipeline import ReasoningPipeline, save_conversation, load_conversation
from cancer_immunology_reasoner.report import format_report


app = typer.Typer(help="Cancer Immunology Causal Reasoning System")
console = Console()


@app.command()
def ingest(
    corpus_dir: Path = typer.Argument(
        Path("data/corpus"), help="Directory containing PDF corpus"
    ),
    output: Path = typer.Option(
        Path("data/index/principles.json"), "--output", "-o",
        help="Output path for extracted principles"
    ),
    skip_llm: bool = typer.Option(
        False, "--skip-llm",
        help="Skip LLM extraction (use existing principles file)"
    ),
):
    """Stage 1: Ingest documents and extract principles."""
    if skip_llm and output.exists():
        console.print("[yellow]Skipping LLM extraction, loading existing principles...[/yellow]")
        with open(output) as f:
            data = json.load(f)
        from cancer_immunology_reasoner.models import Principle
        principles = [Principle(**p) for p in data]
    else:
        console.print("[bold green]Stage 1: Document Ingestion & Principle Extraction[/bold green]")
        console.print(f"Corpus: {corpus_dir}")
        principles = ingest_corpus(corpus_dir, output)
    
    console.print(f"\n[green]OK[/green] Extracted {len(principles)} principles")
    console.print(f"[green]OK[/green] Saved to {output}")
    
    # Build index
    console.print("[bold green]Building vector index and dependency graph...[/bold green]")
    from cancer_immunology_reasoner.retrieval import PrincipleIndex
    index = PrincipleIndex(principles)
    index_path = output.parent / "index.json"
    index.save(index_path)
    console.print(f"[green]OK[/green] Index saved to {index_path}")


@app.command()
def export_web(
    index_path: Path = typer.Argument(
        Path("data/index/index.json"),
        help="Path to the principle index built by ingest"
    ),
    output: Path = typer.Option(
        Path("docs/data/principles.json"), "--output", "-o",
        help="Output path for the browser-friendly principles bundle"
    ),
):
    """Export a browser-friendly principles bundle (no embeddings) for GitHub Pages."""
    if not index_path.exists():
        console.print(f"[red]✗[/red] Index not found at {index_path}")
        console.print("Run 'python -m cancer_immunology_reasoner.cli ingest' first")
        raise typer.Exit(1)

    from cancer_immunology_reasoner.models import Principle
    with open(index_path) as f:
        data = json.load(f)

    principles = []
    for pid, p in data["principles"].items():
        p.pop("embedding", None)
        Principle(**p)
        principles.append(p)

    bundle = {
        "principles": principles,
        "depends_on": data["depends_on"],
        "dependents": data["dependents"],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(bundle, f, indent=2)

    console.print(f"[green]OK[/green] Exported {len(principles)} principles to {output}")


@app.command()
def query(
    text: str = typer.Argument(..., help="Research query"),
    index_path: Path = typer.Option(
        Path("data/index/index.json"), "--index", "-i",
        help="Path to the principle index"
    ),
    conversation_out: Optional[Path] = typer.Option(
        None, "--save", "-s",
        help="Save conversation state to file"
    ),
    refine: Optional[Path] = typer.Option(
        None, "--refine", "-r",
        help="Load previous conversation for refinement"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", help="Enter interactive refinement mode"
    ),
):
    """Run the full query pipeline (Stages 2-6)."""
    from cancer_immunology_reasoner.retrieval import PrincipleIndex
    
    if not index_path.exists():
        console.print(f"[red]✗[/red] Index not found at {index_path}")
        console.print("Run 'python -m cancer_immunology_reasoner.cli ingest' first")
        raise typer.Exit(1)
    
    console.print("[bold green]Cancer Immunology Causal Reasoning System[/bold green]")
    console.print("=" * 60)
    
    pipeline = ReasoningPipeline(index_path)
    
    if refine:
        # Load previous conversation
        from cancer_immunology_reasoner.pipeline import load_conversation
        state = load_conversation(refine)
        console.print(f"[yellow]Refining previous query:[/yellow] {state.query}")
        console.print(f"[yellow]Your new input:[/yellow] {text}")
        report, state = pipeline.refine(state, text)
    else:
        report, state = pipeline.run(text)
    
    # Print report
    formatted = format_report(report)
    console.print(Panel(Markdown(f"```\n{formatted}\n```"), title="Reasoning Report", expand=False))
    
    # Save conversation
    if conversation_out:
        save_conversation(state, conversation_out)
        console.print(f"\n[green]OK[/green] Conversation saved to {conversation_out}")
    
    # Interactive refinement
    if interactive:
        _interactive_loop(pipeline, state)
    
    return state


@app.command()
def refine_conversation(
    conversation_path: Path = typer.Argument(..., help="Path to saved conversation"),
    feedback: str = typer.Argument(..., help="Feedback or additional context"),
    causal_step: Optional[int] = typer.Option(
        None, "--step", help="Disputed causal step number"
    ),
):
    """Stage 7: Refine reasoning based on feedback."""
    from cancer_immunology_reasoner.pipeline import load_conversation, ReasoningPipeline
    
    console.print("[bold green]Stage 7: Feedback Loop[/bold green]")
    
    state = load_conversation(conversation_path)
    console.print(f"[yellow]Original query:[/yellow] {state.query}")
    console.print(f"[yellow]Feedback:[/yellow] {feedback}")
    
    pipeline = ReasoningPipeline(Path("data/index/principles.json"))
    
    target = {}
    if causal_step:
        target["causal_step"] = causal_step
        target["disputed_claim"] = feedback
    else:
        target["additional_context"] = feedback
    
    report, state = pipeline.refine(state, feedback, target)
    
    formatted = format_report(report)
    console.print(Panel(Markdown(f"```\n{formatted}\n```"), title="Refined Report", expand=False))
    
    save_conversation(state, conversation_path)
    console.print(f"[green]OK[/green] Updated conversation saved to {conversation_path}")


def _interactive_loop(pipeline: ReasoningPipeline, state):
    """Interactive refinement loop."""
    console.print("\n[bold]Interactive refinement mode[/bold]")
    console.print("Enter feedback or additional context (or 'exit' to quit)")
    
    while True:
        feedback = Prompt.ask("\n[bold cyan]Your feedback[/bold cyan]")
        if feedback.lower() in ("exit", "quit", "q"):
            break
        
        step = Prompt.ask(
            "[bold cyan]Challenge a specific step? (number, or Enter to skip)[/bold cyan]",
            default=""
        )
        
        target = {}
        if step and step.isdigit():
            target["causal_step"] = int(step)
            target["disputed_claim"] = feedback
        else:
            target["additional_context"] = feedback
        
        with console.status("[bold green]Regenerating reasoning...[/bold green]"):
            report, state = pipeline.refine(state, feedback, target)
        
        formatted = format_report(report)
        console.print(Panel(Markdown(f"```\n{formatted}\n```"), title="Refined Report", expand=False))


if __name__ == "__main__":
    app()