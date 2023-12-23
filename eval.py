# https://github.com/Haskely/gsm8k-rft-llama7b-u13b_evaluation/tree/main
import json
import re
from pathlib import Path
from typing import Callable
from tqdm import tqdm

# Import the solver module
from solver import solver
solve=solver()
def main(
    # gsm8k_test_jsonl: str = "dataset/hard/gsm8k-hard.jsonl",
    gsm8k_test_jsonl: str="new_file.jsonl",
    batch_size: int = 32,
    save_dir: str | None = None,
):
    print(f"main start, batch_size:{batch_size}")
    with open(gsm8k_test_jsonl, "r") as f:
        gsm8k_datas = [json.loads(line) for line in f]  

    # Create the output directory if it does not exist
    if save_dir is None:
        save_dir = f"./output_bs{batch_size}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    gen_datas_jsonl = Path(save_dir) / "gen_datas.jsonl"
    start_index = (
        len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
    )
    print(f"start_index: {start_index}")

    # Modify the loop to use solver.run instead of model prediction
    for i in tqdm(range(start_index, len(gsm8k_datas), batch_size)):
        cur_gsm8k_batch = gsm8k_datas[i : i + batch_size]
        output_str_list = [solve.run_no_agg(d["question"]) for d in tqdm(cur_gsm8k_batch)]

        for j, (gsm8k_data, output_str) in tqdm(enumerate(
            zip(cur_gsm8k_batch, output_str_list)
        )):
            with open(gen_datas_jsonl, "a") as f:
                json.dump(
                    dict(
                        index=i + j,
                        gsm8k_data=gsm8k_data,
                        output_str=output_str,
                    ),
                    f,
                )
                f.write("\n")

    # Calculate accuracy
    with open(gen_datas_jsonl) as f:
        gen_datas = [json.loads(line) for line in f]

    correct_results = []
    wrong_results = []
    for gen in gen_datas:
        result = dict(
            **gen,
            extract_true_num=extract_last_num(gen["gsm8k_data"]["answer"]),
            extract_pred_num=extract_last_num(gen["output_str"]),
            is_correct=None,
        )
        if result["extract_true_num"] is not None and result["extract_pred_num"] is not None:
            if abs(result["extract_true_num"] - result["extract_pred_num"]) < 1e-3:
                result["is_correct"] = True
                correct_results.append(result)
            else:
                result["is_correct"] = False
                wrong_results.append(result)
        else:
            result["is_correct"] = False
            wrong_results.append(result)

    with open(Path(save_dir) / "correct.json", "w") as f:
        json.dump(correct_results, f, ensure_ascii=False, indent=4)
    with open(Path(save_dir) / "wrong.json", "w") as f:
        json.dump(wrong_results, f, ensure_ascii=False, indent=4)

    result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)})={len(correct_results)/(len(correct_results) + len(wrong_results))}"
    print(result)

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # Process numbers like 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # Match numbers like 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return None

if __name__ == "__main__":
    import fire
    fire.Fire(main)
