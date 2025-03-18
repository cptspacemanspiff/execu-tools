import transformers as tr
from transformers.cache_utils import StaticCache
import torch

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

def contrastive_generation(amateur, expert, prompt, max_tokens) -> str:
    amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur)
    expert_model = tr.AutoModelForCausalLM.from_pretrained(expert)

    static_cache_expert = StaticCache(expert_model.config, 1, max_tokens)
    static_cache_amateur = StaticCache(amateur_model.config, 1, max_tokens)

    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

    # expected_output = expert_model.generate(tokens, num_beams=1)

    expert_output = expert_model.forward(
        tokens,
        past_key_values=static_cache_expert,
        use_cache=True,
        num_logits_to_keep=1,
    )
    amateur_output = amateur_model.forward(
        tokens,
        past_key_values=static_cache_expert,
        use_cache=True,
        num_logits_to_keep=1,
    )

    def log_likelyhood(expert_logits, amateur_logits):
        


    out = torch.argmax(expert_output.logits, 2)
    pos = torch.zeros(1, 1, dtype=torch.long)
    pos += tokens.size(1)
    while pos.item() < max_tokens:
        expert_output = expert_model.forward(
            input_ids=out,
            position_ids = pos,
            past_key_values=static_cache_expert,
            num_logits_to_keep=1,
            use_cache=True,
        )
        out = torch.argmax(expert_output.logits, 2)
        pos += 1


if __name__ == "__main__":
    print("main")
    contrastive_generation(amateur_path, expert_path, prompt, 1000)
