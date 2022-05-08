# README
Model of judgment documents reasoning evaluation system.

## Example

(All examples below take the crime of intentionally injuring as example)

- Run flask
    ```text
    python mian.py
    ```

- Train model with BERT
    ```text
    python train_model.py --crime hurt --negative_multiple 5 --embedding_dim 768 --batch_size 16 --epochs 10 --earlystop_patience 3
    ```

- Test checking single judgment document
    ```text
    python test_check.py --crime hurt --txt_name 001
    ```

- Test checking batch judgment documents
    ```text
    python check_tools.py
    ```

## Citation

```text
@ARTICLE{ge2021learning,
    author={Ge, Jidong and Huang, Yunyun and Shen, Xiaoyu and Li, Chuanyi and Hu, Wei},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
    title={Learning Fine-Grained Fact-Article Correspondence in Legal Cases}, 
    year={2021},
    volume={29},
    pages={3694-3706},
    doi={10.1109/TASLP.2021.3130992}
}
```
