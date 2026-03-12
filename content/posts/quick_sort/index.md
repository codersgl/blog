+++
date = '2026-01-13T22:00:47+08:00'
draft = true
title = 'Quick_sort'
+++

最近在刷算法题，碰到了一个自定义排序问题，而且又在学C语言，尽管C标准库的头文件`stdlib.h`中提供了一个快排算法的模板，但我还是想自己实现一下，正好复习一下快排，真忘光了，哈哈。

## 快排的原理

简单来说，快排就只有两个步骤：

1. 选基准：在数组中选择一个数作为基准数。一般选择第一个元素，即`arr[0]`，最优的是随机选择，避免坏情况。
2. 分区：遍历数组，将所有小于等于基准的元素放到基准左边，大于等于基准的放到右边，最终基准会被放到 “排序后它该在的位置”。分区的经典实现（双指针法）：
    1. 定义两个指针：low（指向数组起始）、high（指向数组末尾）；
    2. 从high开始向左找，找到第一个小于基准的元素，停住；
    3. 从low开始向右找，找到第一个大于基准的元素，停住；
    4. 交换这两个元素；
    5. 重复 2-4，直到low >= high，最后将基准和low（或high）位置的元素交换，完成分区。

## 交换

首先实现一个简单的交换算法用来交换两个元素。

```c
void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}
```

## 选取基准值

基准值的选取对算法的性能有很大的影响，业界常用的选数法为**三数取中法**。原理是在左右端点和中间元素的值中选择中位数作为基准值，最后将基准值置于左端点。

```c
int midianOfTree(int arr[], int low, int high)
{
    int mid = low + (high - low) / 2;

    if (arr[low] > arr[mid])
    {
        swap(&arr[low], &arr[mid]);
    }
    if (arr[low] > arr[high])
    {
        swap(&arr[low], &arr[high]);
    }
    if (arr[mid] > arr[high])
    {
        swap(&arr[mid], &arr[high]);
    }

    swap(&arr[mid], &arr[low]);
    return arr[low];
}
```

## 分区

接下来就是分区，采用双指针法，先将从右往左比较元素，找到第一个小于基准值的元素，至于为什么是从右往左开始，而不是从左往右开始，是因为我们设置基准值为左端点的值。然后从左往右，找到第一个大于基准值的元素。然后交换这两个元素。重复上述步骤，直到双指针重合。

最后将基准值移动到合适的位置，即双指针最后停留的位置，这个位置所有左边的元素都比基准值小，所有右边的元素都比基准值大。返回指向基准值的指针。

> 这里有个问题，为什么用`swap(&ar[low], &arr[i])`而不直接用`swap(&pivot, &arr[i])`，是因为左指针的初始为值一定等于`pivot`，因此指针一定会右移，从而在遍历过程中不会修改`arr[low]`，它始终等于`pivot`。



```c
int partition(int arr[], int low, int high)
{
    int pivot = midianOfTree(arr, low, high);
    int i = low;
    int j = high;

    while (low < high)
    {
        while (i < j && arr[j] >= pivot)
        {
            j--;
        }
        while (i < j && arr[i] <= pivot)
        {
            i++;
        }
        swap(&arr[i], &arr[j]);
    }

    swap(&arr[low], &arr[i]);
    return i;
}

```

## 递归排序

先用分区函数分区，再分别对左子区和右子区进行排序，直到子区大小为1。

```c
void quick_sort(int arr[], int low, int high)
{
    if (low < high)
    {
        int pivot_pos = partition(arr, low, high);
        quick_sort(arr, low, pivot_pos - 1);
        quick_sort(arr, pivot_pos + 1, high);
    }
}
```

## 总结

快排算法采用了分而治之的思想，将对一个数组排序分解成对多个子数组排序，大大提高了排序的效率。