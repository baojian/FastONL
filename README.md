# FastONL

Code and datasets for FastONL (ICML, 2023).

To run FastONL on a specific dataset, one can use

```shell
python fastonc.py --kernel=k1 --dataset=citeseer --lambda=1000 --eps=1e-5 --seed=0
```

The output looks like:
```markdown
run on citeseer dataset with 2110 nodes, using kernel k1
accuracy: 0.741 run time: 1.03 seconds
```

To demonstrate our FastONL, we run FastONL on 
citeseer dataset, use the following code
```shell
python demo_fastonc.py --dataset=citeseer
```
The output of the above code could look like:
```markdown
--------------------kernel-1--------------------
lambda: 211.0 , acc: 0.73460, run-time: 11.46 seconds
lambda: 422.0 , acc: 0.73649, run-time: 3.80 seconds
lambda: 633.0 , acc: 0.73886, run-time: 1.96 seconds
lambda: 844.0 , acc: 0.73934, run-time: 1.31 seconds
lambda: 1055.0 , acc: 0.74076, run-time: 0.89 seconds
lambda: 1266.0 , acc: 0.74313, run-time: 0.67 seconds
lambda: 1477.0 , acc: 0.74028, run-time: 0.53 seconds
lambda: 1688.0 , acc: 0.74076, run-time: 0.44 seconds
lambda: 1899.0 , acc: 0.74218, run-time: 0.37 seconds
--------------------kernel-2--------------------
lambda: 211.0 , acc: 0.73649, run-time: 1.21 seconds
lambda: 422.0 , acc: 0.73981, run-time: 0.38 seconds
lambda: 633.0 , acc: 0.73886, run-time: 0.25 seconds
lambda: 844.0 , acc: 0.73555, run-time: 0.19 seconds
lambda: 1055.0 , acc: 0.73081, run-time: 0.16 seconds
lambda: 1266.0 , acc: 0.72464, run-time: 0.14 seconds
lambda: 1477.0 , acc: 0.72322, run-time: 0.12 seconds
lambda: 1688.0 , acc: 0.72038, run-time: 0.11 seconds
lambda: 1899.0 , acc: 0.71801, run-time: 0.11 seconds
--------------------kernel-3--------------------
lambda: 211.0 , acc: 0.67393, run-time: 0.07 seconds
lambda: 422.0 , acc: 0.68720, run-time: 0.08 seconds
lambda: 633.0 , acc: 0.69621, run-time: 0.08 seconds
lambda: 844.0 , acc: 0.70142, run-time: 0.08 seconds
lambda: 1055.0 , acc: 0.70284, run-time: 0.08 seconds
lambda: 1266.0 , acc: 0.70758, run-time: 0.09 seconds
lambda: 1477.0 , acc: 0.70900, run-time: 0.09 seconds
lambda: 1688.0 , acc: 0.71280, run-time: 0.10 seconds
lambda: 1899.0 , acc: 0.71374, run-time: 0.10 seconds
--------------------kernel-4--------------------
lambda: 211.0 , acc: 0.74171, run-time: 0.33 seconds
lambda: 422.0 , acc: 0.74171, run-time: 0.33 seconds
lambda: 633.0 , acc: 0.74171, run-time: 0.33 seconds
lambda: 844.0 , acc: 0.74171, run-time: 0.33 seconds
lambda: 1055.0 , acc: 0.74171, run-time: 0.33 seconds
lambda: 1266.0 , acc: 0.74171, run-time: 0.33 seconds
lambda: 1477.0 , acc: 0.74171, run-time: 0.33 seconds
lambda: 1688.0 , acc: 0.74171, run-time: 0.33 seconds
lambda: 1899.0 , acc: 0.74171, run-time: 0.33 seconds
```

Contact: Baojian Zhou, bjzhou@fudan.edu.cn