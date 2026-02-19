# Can Vision Language Models Infer Human Gaze Direction? A Controlled Study
The public reproducible analysis code used for the project: `Can Vision Language Models Infer Human Gaze Direction? A Controlled Study.`
Large language models are used to assist in coding, but the authors take full responsibility for all content.

[Project Webpage](https://grow-ai-like-a-child.github.io/gaze/) • [Arxiv](https://arxiv.org/abs/2506.05412) • [Preprint PDF](https://grow-ai-like-a-child.github.io/gaze/static/pdfs/paper.pdf) • [Evaluation Stimulus Set](https://osf.io/kyaeu) • [GrowAI Team](https://growing-ai-like-a-child.github.io/)

# File Structure
- [Figure Reproduction] [plot.ipynb](./plot.ipynb) contains the code to reproduce most of the figures in the same order as in the paper, which uses metadata in the `model_info` folder.
- [Power Analysis] [power_analysis.ipynb](./power_analysis.ipynb) performs a post-hoc power analysis to distinguish true null effects from inconclusive insignificant results.
- [Model Fitting and Selection] Fit separate mixed-effects models for each group.
    - [gemini.ipynb](./gemini.ipynb)
    - [glm.ipynb](./glm.ipynb)
    - [gpt.ipynb](./gpt.ipynb)
    - [internlm.ipynb](./internlm.ipynb)
    - [qwen.ipynb](./qwen.ipynb)
    - [humans.ipynb](./humans.ipynb)
    - [moondream.ipynb](./moondream.ipynb)
    - [gazelle.ipynb](./gazelle.ipynb) (models include gazelle_dinov2_vitb14_no_bbox, gazelle_dinov2_vitb14, gazelle_dinov2_vitl14_no_bbox, and gazelle_dinov2_vitl14; "b" for base, "l" for large, "no_bbox" for without face bounding box input)
- [Statistical Visualization] [aggregate.ipynb](./aggregate.ipynb) aggregates the results from the individual models for visualization of estimated marginal means (and trends).
- [Stimuli Metadata] [stimuli_1763865486.csv](./stimuli_1763865486.csv) contains metadata for the stimuli used in the study. Entries with `list_id == -1` are attention checks not used for statistical analysis.
- [All VLM and human responses] [result_1743457603_20250506_20250506F.csv](./result_1743457603_20250506_20250506F.csv) contains all the responses from the VLMs and human participants. TODO

# Preparation
```bash
conda create --yes --name gaze python=3.14
conda activate gaze
python -m pip install tqdm ipython numpy matplotlib pandas ipywidgets ipykernel scipy seaborn
conda install -c conda-forge r-irkernel jupyter r-lme4 r-readr r-marginaleffects r-ggplot2 r-dplyr r-parameters r-performance r-tidyr r-car r-ggeffects
```

# Statistical Notes
To test interaction "Proximity times n_candidates":
```
avg_slopes(model, variables = "Proximity", by = "n_candidates")
avg_slopes(model, variables = "Proximity", by = "n_candidates", hypothesis = ~ pairwise)
```

# Conversion of VLM Outputs to Multiple Choice Answers
```python
import json
import pandas as pd
import string
import re
import argparse
import glob
import os
import tqdm
from match_utils import *
from in_match import match_in
def rm_model_special(pred):
    if '>\n\n' in pred:
        pred = pred.split('>\n\n')[-1]
    if '**\n\n' in pred:
        pred = pred.split('**\n\n')[-1]
    pred = pred.replace("\[ \\boxed{", "")
    pred = pred.replace("} \\]", "")
    pred = pred.replace("<\uff5cend\u2581of\u2581sentence\uff5c>", "")
    pred = pred.replace("<|end_of_sentence|>", "")
    pred = pred.replace("</s>", "")
    pred = pred.replace("<CONCLUSION>", "")
    pred = pred.replace("</CONCLUSION>", "")
    pred = pred.replace("Falcon: ", "")
    pred = pred.strip()
    return pred
## Template Match
def match_template(row):
    pred = row['output']
    pred = str(pred)
    pred = pred.strip()
    res = 'Fail'
    if row['type'] == "NU" and len(pred.split())>2:
        pattern = r'\b(\d+)\b'
        match = re.search(pattern, pred, re.IGNORECASE)
        if match:
            res = match.group().upper()
        else:
            res = "Fail"
    elif row['type'] in ["MC","TF"] and len(pred.split())>=2:
        patterns = [r'^(yes|no|\w)(,|\.|\;| |\n|\*)+',
                                r'[\n\*\{]+(yes|no|\w)(,|\.|\;| |\n|\*|\})+',
                                r'(yes|no|\w) is the correct answer',
                                r'answer is[\:\;\*\n ]*(yes|no|\w)',
                                r'answer[\:\;\*\n ]*(yes|no|\w)',
                                r'choice is[\:\;\*\n ]*(yes|no|\w)',
                                r'choice[\:\;\*\n ]*(yes|no|\w)',
                                r'option is[\:\;\*\n ]*(yes|no|\w)',
                                r'Assistant[\:\;\*\n ]*(yes|no|\w)',
                                ]
        type_ans = {
            'MC': OPTIONS,
            'TF': ['YES','NO']
        }
        for pattern in patterns:
            match = re.search(pattern, pred, re.IGNORECASE)
            if match:
                newres = match.group(1).upper()
                if newres in type_ans[row['type']]:
                    res = newres
                    break
            else:
                res = 'Fail'
    else:
        if pred == "A.":
            res = "A"
        elif pred == "B.":
            res = "B"
        elif pred == "C.":
            res = "C"
        elif pred == "D.":
            res = "D"
        res = re.split(r',|\.| |\:|\;|\n',pred)[0].upper() if len(pred)>0 else pred
    if (res.lower() not in ["yes", "no"] and not re.fullmatch(r"[a-f0-9]", res.lower())):
        return "Fail"
    return res  
def match(infile,outfile):
    # if os.path.exists(outfile):
    #     #infile = outfile
    #     print(f"File already exists {outfile}")
    with open(infile) as f:
        js_file = json.load(f)
    print(f"Loaded {len(js_file)} items from {infile}")
    js_file = [item for item in js_file if isinstance(item, dict)]
    data = pd.DataFrame(js_file)
    data['output'] = data['output'].apply(lambda x: rm_model_special(x))
    data = data.apply(lambda x: format(x), axis=1)
    try:
        data['id'] = (data['id'] %1e6).astype(int) ## in circular eval, same group is indexed as 1,100001,200001..
    except Exception as e:
        print(f"Invalid index in {infile}")
    data['output'] = data['output'].apply(lambda x: rm_special(x))
    data['output'] = data.apply(lambda x: match_template(x),axis=1)
    ## Update matching in json
    for id, sample in enumerate(js_file):
        sample['temp_match'] = data.loc[int(id),'output']
        # sample['stage_lvl'] = data.loc[int(id),'stage_lvl']
        sample['concept_type'] = data.loc[int(id),'concept_type']
    ## dump failed samples to another outfile, add _FAIL to the outfile name
    #if "work_dirs/template_match" in outfile:
    print("Dumping failed results to /template_match_FAIL...")
    outfile_fail = outfile.replace("/template_match","/template_match_FAIL")
    os.makedirs(os.path.dirname(outfile_fail),exist_ok=True)
    js_file_fail = [sample for sample in js_file if sample['temp_match'] == "Fail"]
    with open(outfile_fail,'w') as f:
        json.dump(js_file_fail,f,indent=4)
    for id, sample in enumerate(js_file):       
        #remove these key and values: "questions", "choices","image","prompt","hint","stage_lvl"
        sample.pop('questions', None)
        sample.pop('choices', None)
        sample.pop('image', None)
        sample.pop('prompt', None)
        sample.pop('hint', None)
        sample.pop('stage_lvl', None)
    os.makedirs(os.path.dirname(outfile),exist_ok=True)
    ## updated json
    with open(outfile,'w') as f:
        json.dump(js_file,f,indent=4) 
def parse_args():
    parser = argparse.ArgumentParser()
    # Work Dir
    parser.add_argument('--infile', type=str, default='work_dirs/total/448/idefics2_8b/p0.json', help="Input directory containing json files")
    parser.add_argument('--outfile', type=str, default='work_dirs/cogdevOut/idefics2_8b/p0.json', help="Output base file name type/reso/model/xxxp0.json")
    args = parser.parse_args()
    return args

def build_prompt_mc(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the answer is already a single uppercase or lowercase character in the given options, output the answer directly.\n'
        'If the meaning of all options are significantly different from the answer, output Z.\n'
        'If the answer is random words, noise or gibberish, also output Z.\n'
        'You should output a single uppercase character in the given options (if they are valid options), or Z (if the answer is invalid). \n'
        'You should output ONLY a single uppercase character WITHOUT ANYTHING ELSE.\n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear, B. rabbit, C. cat, D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear, B. rabbit, C. cat, D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: To hang a framed photo on a wall, which tool should you use?\nOptions: A, B, C, D, E\n'
        'Answer: d\nYour output: D\n'
        'Example 4: \n'
        'Question: To hang a framed photo on a wall, which tool should you use?\nOptions: A, B, C, D, E\n'
        'Answer: (empty space)\nYour output: Z\n'
        'Example 5: \n'
        'Question: To hang a framed photo on a wall, which tool should you use?\nOptions: A, B, C, D, E\n'
        'Answer: 0\<<<<<<>\nYour output: Z\n'
        'Example 6: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear, B. rabbit, C. cat, D. dog\n'
        'Answer: B\nYour output: B\n'
        'Example 7: \n'
        'Question: {}\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)

if __name__ == "__main__":
    args = parse_args()
    match(args.infile,args.outfile)
```

# Citation
If you find our work helpful for your research, please give us a star and cite as follows :)
```
@article{vlmGaze2025,
  title={Can Vision Language Models Infer Human Gaze Direction? A Controlled Study},
  author={Zhang, Zory and Feng, Pinyuan and Wang, Bingyang and Zhao, Tianwei and Yu, Suyang and Gao, Qingying and Deng, Hokin and Ma, Ziqiao and Li, Yijiang and Luo, Dezhi},
  year={2025},
  eprint={2506.05412},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.05412},
}
```