  在3的基础上，增加了对图片训练前的处理，将其转换为tfrecords文件。
  同时，在训练、测试的基础上加入了应用，输入图片，输出预测结果。
  运行顺序：1.mnist_generateds.py 2.mnist_backward.py 3.mnist_test.py 4. mnist_app.py