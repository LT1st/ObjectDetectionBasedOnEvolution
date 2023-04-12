--update...
Now, I have release the algorithm into the dictionary "code".


All codes are implemented from the architecture and weight initialization optimized by my own algorithm that will be released soon.

These codes have been tested in the GPU with NVIDIA 1080, and written by Tensorflow1.3.

Feel free to contact me when encouter the propoblems in reproducing the results. <sunkevin1214@gmail.com>


# Steven 2023
# Dataset

# Usage

## code
```
conda create -n tf-gpu tensorflow-gpu

pip install requirement.txt
```
## file
```
│  back_image.py        
│  back_rand.py
│  basic.py                 训练网络代码，基础
│  convex.py
│  evaluate.py
│  evolve.py
│  get_data.py
│  individual.py
│  layers.py
│  main.py
│  MNISTfashion.py
│  population.py
│  README.md
│  rectangle.py
│  rectangle_image.py
│  rot.py
│  rot_back_image.py
│  utils.py
```


# BUG
## 2.28 tooo slow in training step
```python

  def _call_tf_sessionrun(self, options, feed_dict, fetch_list, target_list,
                          run_metadata):
    return tf_session.TF_SessionRun_wrapper(
        self._session, options, feed_dict, fetch_list, target_list,
        run_metadata)
```

The data seems to be placed in CPU and the GPU memory keeps to be free all the time.


## 2.28.2 exit code 143
```angular2html
Process finished with exit code 143
```

## inf grad
```
2023-02-28 15:45:42.428397: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0)- -> (device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:3b:00.0)
2023-02-28 15:45:45.993568: E tensorflow/core/kernels/check_numerics_op.cc:157] abnormal_detected_host @0x7fe4c69bb300 = {1, 0} LossTensor is inf or nan
2023-02-28 15:45:45.993760: W tensorflow/core/framework/op_kernel.cc:1192] Invalid argument: LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
2023-02-28 15:45:46.043692: W tensorflow/core/framework/op_kernel.cc:1192] Invalid argument: LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
	 [[Node: I_0_test/Mean/_127 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_342_I_0_test/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

Caused by op 'I_0_train/train_op/CheckNumerics', defined at:
  File "/tmp/pycharm_project_340/main.py", line 86, in <module>
    begin_evolve(0.9, 0.05, 0.2, 0.05, pop_size, train_images, train_label, test_images, test_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, total_generation_number, eta)
  File "/tmp/pycharm_project_340/main.py", line 23, in begin_evolve
    cnn.evaluate_fitness(0)
  File "/tmp/pycharm_project_340/evolve.py", line 70, in evaluate_fitness
    evaluate.parse_population(gen_no)
  File "/tmp/pycharm_project_340/evaluate.py", line 41, in parse_population
    rs_mean, rs_std, num_connections, new_best = self.parse_individual(indi, self.number_of_channel, i, save_dir, history_best_score)
  File "/tmp/pycharm_project_340/evaluate.py", line 154, in parse_individual
    = self.build_graph(indi_index, num_of_input_channel, indi, train_data, train_label, validate_data, validate_label)
  File "/tmp/pycharm_project_340/evaluate.py", line 137, in build_graph
    train_op = slim.learning.create_train_op(cross_entropy, optimizer)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/contrib/slim/python/slim/learning.py", line 440, in create_train_op
    check_numerics=check_numerics)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/contrib/training/python/training/training.py", line 456, in create_train_op
    'LossTensor is inf or nan')
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/ops/gen_array_ops.py", line 413, in check_numerics
    message=message, name=name)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
    op_def=op_def)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 2630, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1204, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
	 [[Node: I_0_test/Mean/_127 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_342_I_0_test/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

finally...
Traceback (most recent call last):
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1327, in _do_call
    return fn(*args)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1306, in _run_fn
    status, run_metadata)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/contextlib.py", line 88, in __exit__
    next(self.gen)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py", line 466, in raise_exception_on_not_ok_status
    pywrap_tensorflow.TF_GetCode(status))
tensorflow.python.framework.errors_impl.InvalidArgumentError: LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
	 [[Node: I_0_test/Mean/_127 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_342_I_0_test/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/tmp/pycharm_project_340/main.py", line 86, in <module>
    begin_evolve(0.9, 0.05, 0.2, 0.05, pop_size, train_images, train_label, test_images, test_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, total_generation_number, eta)
  File "/tmp/pycharm_project_340/main.py", line 23, in begin_evolve
    cnn.evaluate_fitness(0)
  File "/tmp/pycharm_project_340/evolve.py", line 70, in evaluate_fitness
    evaluate.parse_population(gen_no)
  File "/tmp/pycharm_project_340/evaluate.py", line 41, in parse_population
    rs_mean, rs_std, num_connections, new_best = self.parse_individual(indi, self.number_of_channel, i, save_dir, history_best_score)
  File "/tmp/pycharm_project_340/evaluate.py", line 209, in parse_individual
    coord.join(threads)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/training/coordinator.py", line 389, in join
    six.reraise(*self._exc_info_to_raise)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/six.py", line 719, in reraise
    raise value
  File "/tmp/pycharm_project_340/evaluate.py", line 169, in parse_individual
    _, accuracy_str, loss_str, _ = sess.run([train_op, accuracy,cross_entropy, merge_summary], {is_training:True})
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 895, in run
    run_metadata_ptr)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1124, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1321, in _do_run
    options, run_metadata)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1340, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
	 [[Node: I_0_test/Mean/_127 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_342_I_0_test/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

Caused by op 'I_0_train/train_op/CheckNumerics', defined at:
  File "/tmp/pycharm_project_340/main.py", line 86, in <module>
    begin_evolve(0.9, 0.05, 0.2, 0.05, pop_size, train_images, train_label, test_images, test_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, total_generation_number, eta)
  File "/tmp/pycharm_project_340/main.py", line 23, in begin_evolve
    cnn.evaluate_fitness(0)
  File "/tmp/pycharm_project_340/evolve.py", line 70, in evaluate_fitness
    evaluate.parse_population(gen_no)
  File "/tmp/pycharm_project_340/evaluate.py", line 41, in parse_population
    rs_mean, rs_std, num_connections, new_best = self.parse_individual(indi, self.number_of_channel, i, save_dir, history_best_score)
  File "/tmp/pycharm_project_340/evaluate.py", line 154, in parse_individual
    = self.build_graph(indi_index, num_of_input_channel, indi, train_data, train_label, validate_data, validate_label)
  File "/tmp/pycharm_project_340/evaluate.py", line 137, in build_graph
    train_op = slim.learning.create_train_op(cross_entropy, optimizer)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/contrib/slim/python/slim/learning.py", line 440, in create_train_op
    check_numerics=check_numerics)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/contrib/training/python/training/training.py", line 456, in create_train_op
    'LossTensor is inf or nan')
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/ops/gen_array_ops.py", line 413, in check_numerics
    message=message, name=name)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
    op_def=op_def)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 2630, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1204, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
	 [[Node: I_0_test/Mean/_127 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_342_I_0_test/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]


Process finished with exit code 1

```

- Filed:  限制梯度
- 尝试归一化
- 降低学习率

```
/home/lutao/.conda/envs/tf-gpu/bin/python /tmp/pycharm_project_706/main.py 
/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:458: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:459: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:460: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:461: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:462: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:465: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
initializing population with number 50...
evaluate fintesss
2023-03-28 21:29:09.760398: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2023-03-28 21:29:09.760431: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2023-03-28 21:29:09.760438: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2023-03-28 21:29:09.760444: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2023-03-28 21:29:09.760449: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX512F instructions, but these are available on your machine and could speed up CPU computations.
2023-03-28 21:29:09.760454: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2023-03-28 21:29:10.054717: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: NVIDIA GeForce RTX 3090
major: 8 minor: 6 memoryClockRate (GHz) 1.695
pciBusID 0000:3b:00.0
Total memory: 23.69GiB
Free memory: 23.43GiB
2023-03-28 21:29:10.054748: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2023-03-28 21:29:10.054755: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2023-03-28 21:29:10.054766: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:3b:00.0)
2023-03-28 21:29:14.445682: E tensorflow/core/kernels/check_numerics_op.cc:157] abnormal_detected_host @0x7fceca256c00 = {1, 0} LossTensor is inf or nan
2023-03-28 21:29:14.445880: W tensorflow/core/framework/op_kernel.cc:1192] Invalid argument: LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
2023-03-28 21:29:14.544756: W tensorflow/core/framework/op_kernel.cc:1192] Invalid argument: LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
	 [[Node: I_0_test/Mean/_179 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_615_I_0_test/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

Caused by op 'I_0_train/train_op/CheckNumerics', defined at:
  File "/tmp/pycharm_project_706/main.py", line 91, in <module>
    begin_evolve(0.9, 0.05, 0.2, 0.05, pop_size, train_images, train_label, test_images, test_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, total_generation_number, eta)
  File "/tmp/pycharm_project_706/main.py", line 27, in begin_evolve
    cnn.evaluate_fitness(0)     # 参数为什么写死了？？？？？？？？？
  File "/tmp/pycharm_project_706/evolve.py", line 73, in evaluate_fitness
    evaluate.parse_population(gen_no)
  File "/tmp/pycharm_project_706/evaluate.py", line 45, in parse_population
    rs_mean, rs_std, num_connections, new_best = self.parse_individual(indi, self.number_of_channel, i, save_dir, history_best_score)
  File "/tmp/pycharm_project_706/evaluate.py", line 160, in parse_individual
    = self.build_graph(indi_index, num_of_input_channel, indi, train_data, train_label, validate_data, validate_label)
  File "/tmp/pycharm_project_706/evaluate.py", line 142, in build_graph
    train_op = slim.learning.create_train_op(cross_entropy, optimizer)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/contrib/slim/python/slim/learning.py", line 440, in create_train_op
    check_numerics=check_numerics)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/contrib/training/python/training/training.py", line 456, in create_train_op
    'LossTensor is inf or nan')
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/ops/gen_array_ops.py", line 413, in check_numerics
    message=message, name=name)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
    op_def=op_def)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 2630, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1204, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
	 [[Node: I_0_test/Mean/_179 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_615_I_0_test/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

finally...
Traceback (most recent call last):
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1327, in _do_call
    return fn(*args)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1306, in _run_fn
    status, run_metadata)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/contextlib.py", line 88, in __exit__
    next(self.gen)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py", line 466, in raise_exception_on_not_ok_status
    pywrap_tensorflow.TF_GetCode(status))
tensorflow.python.framework.errors_impl.InvalidArgumentError: LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
	 [[Node: I_0_test/Mean/_179 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_615_I_0_test/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/tmp/pycharm_project_706/main.py", line 91, in <module>
    begin_evolve(0.9, 0.05, 0.2, 0.05, pop_size, train_images, train_label, test_images, test_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, total_generation_number, eta)
  File "/tmp/pycharm_project_706/main.py", line 27, in begin_evolve
    cnn.evaluate_fitness(0)     # 参数为什么写死了？？？？？？？？？
  File "/tmp/pycharm_project_706/evolve.py", line 73, in evaluate_fitness
    evaluate.parse_population(gen_no)
  File "/tmp/pycharm_project_706/evaluate.py", line 45, in parse_population
    rs_mean, rs_std, num_connections, new_best = self.parse_individual(indi, self.number_of_channel, i, save_dir, history_best_score)
  File "/tmp/pycharm_project_706/evaluate.py", line 215, in parse_individual
    coord.join(threads)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/training/coordinator.py", line 389, in join
    six.reraise(*self._exc_info_to_raise)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/six.py", line 719, in reraise
    raise value
  File "/tmp/pycharm_project_706/evaluate.py", line 175, in parse_individual
    _, accuracy_str, loss_str, _ = sess.run([train_op, accuracy,cross_entropy, merge_summary], {is_training:True})
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 895, in run
    run_metadata_ptr)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1124, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1321, in _do_run
    options, run_metadata)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1340, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
	 [[Node: I_0_test/Mean/_179 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_615_I_0_test/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

Caused by op 'I_0_train/train_op/CheckNumerics', defined at:
  File "/tmp/pycharm_project_706/main.py", line 91, in <module>
    begin_evolve(0.9, 0.05, 0.2, 0.05, pop_size, train_images, train_label, test_images, test_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, total_generation_number, eta)
  File "/tmp/pycharm_project_706/main.py", line 27, in begin_evolve
    cnn.evaluate_fitness(0)     # 参数为什么写死了？？？？？？？？？
  File "/tmp/pycharm_project_706/evolve.py", line 73, in evaluate_fitness
    evaluate.parse_population(gen_no)
  File "/tmp/pycharm_project_706/evaluate.py", line 45, in parse_population
    rs_mean, rs_std, num_connections, new_best = self.parse_individual(indi, self.number_of_channel, i, save_dir, history_best_score)
  File "/tmp/pycharm_project_706/evaluate.py", line 160, in parse_individual
    = self.build_graph(indi_index, num_of_input_channel, indi, train_data, train_label, validate_data, validate_label)
  File "/tmp/pycharm_project_706/evaluate.py", line 142, in build_graph
    train_op = slim.learning.create_train_op(cross_entropy, optimizer)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/contrib/slim/python/slim/learning.py", line 440, in create_train_op
    check_numerics=check_numerics)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/contrib/training/python/training/training.py", line 456, in create_train_op
    'LossTensor is inf or nan')
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/ops/gen_array_ops.py", line 413, in check_numerics
    message=message, name=name)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
    op_def=op_def)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 2630, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/home/lutao/.conda/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1204, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): LossTensor is inf or nan : Tensor had NaN values
	 [[Node: I_0_train/train_op/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="LossTensor is inf or nan", _device="/job:localhost/replica:0/task:0/gpu:0"](I_0_train/control_dependency_1)]]
	 [[Node: I_0_test/Mean/_179 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_615_I_0_test/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]


Process finished with exit code 1

```