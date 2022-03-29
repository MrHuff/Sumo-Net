# SuMo-net

This is the code repository for Sumo-net proposed in the paper:

"Survival Regression with Proper Scoring Rules and Monotonic Neural Networks".

For a quick background/presentation, we refer to our talk at AISTATS 2022: 



To quickly run a model on a dataset, install dependencies, fill out the parameters in debug_run.py and simply run the file for a test run.

To install as a package, do:
```
git clone https://github.com/MrHuff/Sumo-Net.git
python setup.py bdist_wheel
pip install /dist/SuMo_net-0.1-py3-none-any.whl
```

To cite:
```
@misc{https://doi.org/10.48550/arxiv.2103.14755,
  doi = {10.48550/ARXIV.2103.14755},
  
  url = {https://arxiv.org/abs/2103.14755},
  
  author = {Rindt, David and Hu, Robert and Steinsaltz, David and Sejdinovic, Dino},
  
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Survival Regression with Proper Scoring Rules and Monotonic Neural Networks},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```
This library is occasionally (as in almost never) maintained by Rob (MrHuff). Please raise an issue if there are any bugs, and it'll hopefully be taken care off.

If it's really urgent (i.e. your experiments aren't working or you are deploying something in healthcare), you can reach me at:

robert (dot) hu (at) stats (dot) ox (dot) ac (dot) uk

