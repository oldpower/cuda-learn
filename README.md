# cuda-learn

## cmake和cuda
```cmake
# 设置算力架构
set(CMAKE_CUDA_ARCHITECTURES "86")
```


## git
### 只添加被修改和删除的文件
```bash
git add -u
```
### git将凭证存在缓存中
```bash
git config --global credential.helper "cache --timeout=3600"
```

### git将凭证存在文件中
```bash
git config --global credential.helper store
```

### 取消缓存
```bash
git config --global --unset credential.helper
```

### 查看当前设置
```bash
git config --global --get credential.helper
```






