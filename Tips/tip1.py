# coding: utf-8
from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np
import pandas as pd


def readEvent(event_path, scalarName):
    '''
        读tensorboard生成的event文件中指定的标量值
            event_path:event文件路径
            scalarName：要操作的标量名称
    '''
    event = event_accumulator.EventAccumulator(event_path)
    event.Reload()
    print("\033[1;34m数据标签：\033[0m")
    print(event.Tags())
    print("\033[1;34m标量数据关键词：\033[0m")
    print(event.scalars.Keys())
    value = event.scalars.Items(scalarName)
    print("你要操作的scalar是：", scalarName)
    return value


def exportToexcel(scalarNameList, excelName):
    '''
        将不同的标量数据导入到同一个excel中，放置在不同的sheet下
            注：excel中sheet名称的命名不能有：/\?*这些符号
    '''
    writer = pd.ExcelWriter(excelName)
    for i in range(len(scalarNameList)):
        scalarName = scalarNameList[i]
        scalarValue = readEvent(event_path, scalarName)
        data = pd.DataFrame(scalarValue)
        if scalarName == 'Train/cls_loss':
            scalarName = 'Train_loss'
        if scalarName == 'Test/cls_loss':
            scalarName = 'Test_loss'
        data.to_excel(writer, sheet_name=scalarName)
    writer.save()
    print("数据保存成功")


def main(dict, scalarNameList, excelName):
    writer = pd.ExcelWriter(excelName)
    for root, dirs, files in os.walk('./all'):
        for file in files:
            filepath = os.path.join(root, file)
            for i in range(len(scalarNameList)):
                scalarName = scalarNameList[i]
                scalarValue = readEvent(filepath, scalarName)
                tt = root.split('/')[2].split('_')[2]
                x = pd.DataFrame(pd.DataFrame(scalarValue)['value']).rename(columns={'value': str(tt)})
                dict[scalarNameList[i]] = pd.concat(
                    [dict[scalarNameList[i]], x], axis=1)
                readEvent(filepath, scalarName)
    # root='./all'
    # list=os.listdir('./all')
    # for i in range(0,len(list)):
    #     path=os.path.join(root,list[i])
    #         for i in range(len(scalarNameList)):
    #             scalarName = scalarNameList[i]
    #             import pdb;pdb.set_trace()
    #             scalarValue = readEvent(root + file, scalarName)
    #             dict[scalarNameList[i]] = pd.concat([dict[scalarNameList[i]], pd.DataFrame(scalarValue)])
    #             readEvent(root + file)

    for k in dict.keys():
        import pdb;
        pdb.set_trace()
        tt = k.replace('/', '_')
        dict[k].to_excel(writer, sheet_name=tt)
    writer.save()


if __name__ == "__main__":
    event_path = "J:/temp/all"
    scalarNameList = ['Train/cls_loss', 'Test/cls_loss', 'Test/top_1_accuracy', 'Valid/cls_loss',
                      'Valid/top_1_accuracy']
    dict = {}
    for i in range(len(scalarNameList)):
        dict[scalarNameList[i]] = pd.DataFrame()
    excelName = "data2.xlsx"
    main(dict, scalarNameList, excelName)
