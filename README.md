# Audio Enhancement using Audio Separator

This repository aims to improve audio recording process on certain devices which tend to capture background noises/typing sound/disturbances from the surrounding.

The enhanced sound will appear as enahnced_audio.wav on the same directory as 

## Benchmarking Command

To start the benchmarking process, clone this repository and use the command below:

```bash
python3 run.py --maps_dir /path/to/maps/dir/
```

The 3 AUC metric scores is calculated by default and can be disabled if required. Refer to the below arguments to run a custom benchmarking test.

```
--find_AUCJ (Default = True)
--find_AUCB (Default = True)
--find_shuffled_AUC (Default = True)
--find_KLdiv (Default = False)
--find_CC (Default = False)
--M_value (Default = 10)
--add_jitter (Default = True)
```

M_value is the number of random fixation maps used to find AUC shuffled
add_jitter adds a small noise to each pixel of saliency map to prevent duplicate values in saliency maps if there are any.

## Conclusion

The following metrics are crucial in benchmarking saliency models, and more benchmarking metrics will be added in the future.