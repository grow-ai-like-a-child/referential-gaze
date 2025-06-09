# Can Vision Language Models Infer Human Gaze Direction? A Controlled Study
The public reproducible analysis code used for the project: `Can Vision Language Models Infer Human Gaze Direction? A Controlled Study.`

[Project Webpage](https://grow-ai-like-a-child.github.io/gaze/) • [Arxiv TODO]() • [Paper PDF](https://grow-ai-like-a-child.github.io/gaze/static/pdfs/paper.pdf) • [Stimuli](https://osf.io/kyaeu)

# File Structure
- [Figure Reproduction] [plot.ipynb](./plot.ipynb) contains the code to reproduce most of the figures in the same order as in the paper, which uses metadata in the `model_info` folder.
- [Model Fitting and Selection] [gemini.ipynb](./gemini.ipynb), [glm.ipynb](./glm.ipynb), [gpt.ipynb](./gpt.ipynb), [internlm.ipynb](./internlm.ipynb), [qwen.ipynb](./qwen.ipynb), and [human.ipynb](./human.ipynb) fit separate mixed-effects models for each group.
- [Statistical Visualization] [aggregate.ipynb](./aggregate.ipynb) aggregates the results from the individual models for visualization of estimated marginal means (and trends).
- [Power Analysis] [power_analysis.ipynb](./power_analysis.ipynb) performs a post-hoc power analysis to distinguish true null effects from inconclusive insignificant results.
- [Stimuli Metadata] [stimuli_1743457603.csv](./stimuli_1743457603.csv) contains metadata for the stimuli used in the study. Entries with `list_id == -1` are attention checks not used for statistical analysis.
- [All VLM and human responses] [result_1743457603_20250506_20250506F.csv](./result_1743457603_20250506_20250506F.csv) contains all the responses from the VLMs and human participants.

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