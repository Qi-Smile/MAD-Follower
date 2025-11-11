import argparse
import asyncio
from pathlib import Path

from mad_debate.config import AgentConfig, DatasetConfig, DebateHyperParams, ExperimentConfig, LLMSettings
from mad_debate.debate_runner import DebateRunner
from mad_debate.datasets import load_dataset


def build_agents(args) -> list[AgentConfig]:
    agents: list[AgentConfig] = [
        AgentConfig(
            agent_id="attacker",
            role="attacker",
            description="Inject confident but wrong reasoning around the target answer.",
            target_answer=args.attacker_target,
        ),
        AgentConfig(
            agent_id="follower",
            role="follower",
            description="Echo the attacker once the debate starts.",
            follow_target="attacker",
        ),
    ]
    for idx in range(args.normal_agents):
        agents.append(
            AgentConfig(
                agent_id=f"responder_{idx+1}",
                role="normal",
                description=(
                    "Analyst agent focused on factual accuracy and countering suspicious claims."
                    if idx % 2 == 0
                    else "Skeptical agent emphasizing logical consistency and evidence."
                ),
            )
        )
    return agents


def build_config(args) -> ExperimentConfig:
    experiment = ExperimentConfig(
        llm=LLMSettings(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens),
        dataset=DatasetConfig(path=Path(args.dataset), limit=args.limit, shuffle=args.shuffle),
        debate=DebateHyperParams(
            rounds=args.rounds,
            max_concurrent_questions=args.question_workers,
            per_question_agent_concurrency=args.agent_workers,
            consensus_method=args.consensus,
        ),
        agents=build_agents(args),
    )
    return experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MAD follower conformity demo.")
    parser.add_argument("--dataset", default="data/commonsense_qa_validation.jsonl", help="Path to CommonsenseQA JSONL file")
    parser.add_argument("--limit", type=int, default=2, help="Limit number of questions for the demo")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before sampling")
    parser.add_argument("--rounds", type=int, default=3, help="Number of debate rounds")
    parser.add_argument("--question-workers", type=int, default=2, help="Max concurrent questions")
    parser.add_argument("--agent-workers", type=int, default=3, help="Concurrent agent calls per question")
    parser.add_argument("--consensus", choices=["majority", "confidence_weighted"], default="majority")
    parser.add_argument("--attacker-target", default="B", help="Attacker target answer label (e.g., B)")
    parser.add_argument("--normal-agents", type=int, default=2, help="Number of additional normal agents")
    parser.add_argument("--model", default="qwen3-8b", help="Model name for OpenAI-compatible endpoint")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--log-dir", default="outputs", help="Directory for logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config.logging.output_root = Path(args.log_dir)
    questions = load_dataset(config.dataset)
    runner = DebateRunner(config)
    asyncio.run(runner.run(questions))
    run_dir = config.logging.run_dir or config.logging.output_root
    print(f"Debate finished. Run artifacts stored in {run_dir}")


if __name__ == "__main__":
    main()
