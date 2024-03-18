Our tools
---

## Extract pcd from rosbag

Please follow [this section](../README.md#data-creation) to extract pcd from rosbag, kitti format, argoverse 2 format, or other format. 

## Label data

Qingwen [fork the point_labeler tool](https://github.com/Kin-Zhang/point_labeler) and rewrite to our own version with dynamic map benchmark format. You should follow the extraction steps first.

Here is the video tutorial for the tool:

- [中文-bilibili]()
- [English-YouTube]()

## Extract ground truth

After label data, you will get a `labels` folder which contains label as semanticKITTI format. Running following script, you will get ground truth to evaluate afterward for DynamicMap Benchmark. Please contribute to this community by sharing your ground truth if it's possible.

```bash
python scripts/py/data/merge_label.py --data_folder /home/kin/data/semindoor
python scripts/py/data/merge_label.py --data_folder /home/kin/data/av2
```

## Evaluate

Please follow [this section](../README.md#evaluate) to evaluate the result.

![](../../assets/imgs/eval_demo.png)