[OK] 256 그레이 LUT CSV 저장: D:/00 업무/00 가상화기술/00 색시야각 보상 최적화/VAC algorithm/VAC_Optimization_Project/data/수작업보정LUT/LUT_2_256_55gray 수정_2.csv
Traceback (most recent call last):
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\data_preparation\x_generation\interpolate_lut_j_to_4096.py", line 252, in <module>
    main()
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\data_preparation\x_generation\interpolate_lut_j_to_4096.py", line 238, in main
    write_LUT_data(f, save_csv)
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\data_preparation\x_generation\interpolate_lut_j_to_4096.py", line 180, in write_LUT_data
    reshaped_data = np.reshape(data, (256, 16)).tolist()
  File "<__array_function__ internals>", line 200, in reshape
  File "C:\python310\lib\site-packages\numpy\core\fromnumeric.py", line 298, in reshape
    return _wrapfunc(a, 'reshape', newshape, order=order)
  File "C:\python310\lib\site-packages\numpy\core\fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: cannot reshape array of size 256 into shape (256,16)
