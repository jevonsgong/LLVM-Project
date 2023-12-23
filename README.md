# LLVM-Project

This is our updated InterCode repository. All codes except our proposed prompting method are in the master branch.

Running requires Docker installed and a ChatGPT api key. Note this updated repo requires an old version of openai==0.28. (Probably can use openai migrate after installation)

Sample running script: python experiments/eval_n_turn.py --data_path "./data/nl2bash/nl2bash_fs_1.json" --dialogue_limit 5 --env "bash" --image_name "intercode-nl2bash" --log_dir "logs/experiments" --max_turns 5 --policy "chat" --template "v2" --model "gpt-3.5-turbo" 
