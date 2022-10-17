# SCN

# Abstract
To infer unknown remote sensing scenarios, most existing technologies use a supervised learning paradigm to train deep neural network (DNN) models on closed datasets. This paradigm faces challenges such as highly spatiotemporal variants and ever-changing scale-heterogeneous remote sensing scenarios. Additionally, DNN models cannot scale to new scenarios. Lifelong learning is an effective solution to these problems. Current lifelong learning approaches focus on overcoming the catastrophic forgetting issue (i.e., a successive increase in heterogeneous remote sensing scenes causes models to forget historical scenes) while ignoring the knowledge recall issue (i.e., how to facilitate the learning of new scenes by recalling historical experiences), which is a significant problem. This article proposes a lifelong learning framework called asymmetric collaborative network (SCN) for lifelong remote sensing image (RSI) classification. This framework consists of two structurally distinct networks: a preserving network (Pres-Net) and a transient network (Trans-Net), which imitate the long- and short-term memory processes in the brain, respectively. Moreover, this framework is based on two synergistic knowledge transfer mechanisms: triple distillation and prior feature fusion. The triple distillation mechanism enables knowledge persistence from Trans-Net to Pres-Net to achieve better memorization; the prior feature fusion mechanism enables knowledge transfer from Pres-Net to Trans-Net to achieve better recall. Experiments on three open datasets demonstrate the effectiveness of SCN for three-, six-, and nine-task-length learning. The idea of asymmetric separation networks and the synergistic strategy proposed in this article are expected to provide new solutions to the translatability of the classification of RSIs in real-world scenarios. The source codes are available at: https://github.com/GeoX-Lab/SCN.


# Citation
If our repo is useful to you, please cite our published paper as follow:

```
Bibtex
@article{Ye2022SCN,
    title={Better Memorization, Better Recall: A Lifelong Learning Framework for Remote Sensing Image Scene Classification},
    author={Ye, Dingqi and Peng, Jian and Li, Haifeng and Bruzzone, Lorenzo},   
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    DOI = {10.1109/TGRS.2022.3190392},
    year={2022},
    type = {Journal Article}
}

Endnote
%0 Journal Article
%A Ye, Dingqi
%A Peng, Jian
%A Haifeng, Li
%A Bruzzone, Lorenzo
%D 2022
%T Better Memorization, Better Recall: A Lifelong Learning Framework for Remote Sensing Image Scene Classification
%B IEEE Transactions on Intelligent Transportation Systems
%R DOI:10.1109/TGRS.2022.3190392
%! Better Memorization, Better Recall: A Lifelong Learning Framework for Remote Sensing Image Scene Classification
```
