# coding= utf-8
import ctypes
import re
import threading
import time
import os
import subprocess
import string
import csv
import struct
import binascii
import sys

import subprocess

from copy import deepcopy

from queue import Queue, LifoQueue, PriorityQueue
import _thread

# pip install pyinstaller -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com 代理下载pip模块

csvFilePath = ''

csvFileHead = [['dateTime', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z',
                'bodySurface_temp', 'ambient_temp', 'hr_raw', 'hr', 'remarks']]
dfCsvFileHeadDict_ = {'dateTime': '', 'acc_x': '', 'acc_y': '', 'acc_z': '', 'gyr_x': '', 'gyr_y': '', 'gyr_z': '',
                      'bodySurface_temp': '', 'ambient_temp': '', 'hr_raw': '', 'hr': '', 'remarks': ''}

s = struct.Struct('III')

PACKAGE_HEADER_RECOGNITION_STRING = 'MDTCPACK'
FILE_HEADER_RECOGNITION_STRING = 'MDTC'

PACKAGE_HEARD_KEY = bytes(PACKAGE_HEADER_RECOGNITION_STRING, encoding='utf-8')

QUEUE_ONE_ITERM_BYTE_SIZE = 1024 * 10
REMARKES_SIZE = 512

# 解决鼠标点击命令行窗口程序停止问题
kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)
# -------------------------------


def debugInfo(string):
    # print(string)
    pass


def csv_write_heard(path, csvData):
    with open(path, 'a+', encoding='utf-8_sig', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        for row in csvData:
            writer.writerow(row)
    return True
# 直接字典字段写入到CSV文件中


def csv_write_dict(path, csvHeard, csvData):
    with open(path, 'a+', encoding='utf-8', newline='')as f:
        writer = csv.DictWriter(f, fieldnames=csvHeard)
        # writer.writeheader()
        writer.writerows(csvData)


def csv_file_remove(path):
    if os.path.exists(path):  # 如果文件存在
        os.remove(path)
    else:
        debugInfo('no such file:%s' % path)  # 则返回文件不存在


def calcAccGryro(value, range):
    if value > 0:
        Denominator = 0x7fff
    else:
        Denominator = 0x8000
    calc = value * range / Denominator
    return format(calc, '.8f')


def read_one_package_raw_date(allFileDataBuff:bytearray, onePackageData, readFileQueue):
    startRecnizOffset = -1
    endRecnizOffset = -1
    while(1):
        try:
            if readFileQueue != '':
                readData = readFileQueue.get(True, 1)
                allFileDataBuff.extend(readData)
                getDataLen = len(readData)
            else:
                getDataLen = 0
        except:
            readFileQueue = ''
            getDataLen = 0

        if startRecnizOffset == -1:
            try:
                startRecnizOffset = allFileDataBuff.index(
                    PACKAGE_HEARD_KEY)
            except:
                break
        try:
            endRecnizOffset = allFileDataBuff.index(
                PACKAGE_HEARD_KEY, 1)
            onePackageData.extend(
                allFileDataBuff[0:endRecnizOffset])
            debugInfo('endRecnizOffset:'+str(endRecnizOffset) +
                      ' allFileDataBuff size:'+str(len(allFileDataBuff)))
            return endRecnizOffset
        except:
            if getDataLen == 0:
                onePackageData.extend(allFileDataBuff[0:])
                debugInfo('getDataLen == 0')
                break
            else:
                continue


def bin2csv(bin_file, csv_file):  # 建立一个任务线程类:  # 在启动线程后任务从这个函数里面开始执行
    tempTimesStamp = 0
    csvFileHeadDict = dfCsvFileHeadDict_
    onePackageData = bytearray()
    allFileDataBuff = bytearray()
    readFileQueue = Queue(maxsize=0)
    debugInfo('saveFile:'+csv_file)
    if os.path.exists(bin_file) == False:
        print('bin2csv: ' + bin_file + ': No such file or directory')
        return 1
    _thread.start_new_thread(ReadFileThread, (bin_file, readFileQueue))
    time.sleep(1)
    if readFileQueue.empty() == True:
        debugInfo('file queue is empty')
        return 1
    csv_file_remove(csv_file)
    csv_write_heard(csv_file, csvFileHead)

    startReadQueueCount = QUEUE_ONE_ITERM_BYTE_SIZE
    while True:
        allFileDataBuff.extend(readFileQueue.get(True, 5))
        if startReadQueueCount > REMARKES_SIZE:
            break
        startReadQueueCount += QUEUE_ONE_ITERM_BYTE_SIZE
    # 解析remarkes
    remarkesString = ''
    try:
        sFileRemarkes = struct.Struct(str(REMARKES_SIZE)+'s')
        (remarkes,) = sFileRemarkes.unpack(
            allFileDataBuff[0:sFileRemarkes.size])
        allFileDataBuff = allFileDataBuff[sFileRemarkes.size:]
        remarkesString = remarkes.decode('utf-8', 'ignore')            
        if '\0' in remarkesString:
            findEnd = remarkesString.index('\0')
            remarkesString = remarkesString[0:findEnd]
        else:
            remarkesString += '\0'
        debugInfo('remarkes:'+remarkesString)
    except:
        debugInfo('unpack remarkes faile')
        return 1
    # saveFileRemark = []
    # saveFileRemark.append(csvFileHeadDict)
    # saveFileRemark[0]['remarks'] = str(remarkes)
    # csv_write_dict(csv_file, csvFileHead[0], saveFileRemark)
    # 解析头
    try:
        sFileHeader = struct.Struct('4sIHH')
        (headerRecogni, headerPackeNum, accRange, gyroRange) = sFileHeader.unpack(
            allFileDataBuff[0:sFileHeader.size])
    except:
        return 1
    allFileDataBuff = allFileDataBuff[sFileHeader.size:]
    debugInfo('headerRecogni:'+headerRecogni.decode('utf-8', 'ignore'))
    debugInfo('headerPackeNum:'+str(headerPackeNum))
    if headerRecogni.decode('utf-8', 'ignore') != FILE_HEADER_RECOGNITION_STRING:
        debugInfo('headerRecogni != FILE_HEADER_RECOGNITION_STRING')
        time.sleep(0.5)
        return 1
    percentCount = 0
    lastPercent = 0
    temptemp = None
    for j in range(0, headerPackeNum):
        # 读取一包数据
        onePackageData = bytearray()
        debugInfo('PackeNum:'+str(j))
        cutSize = read_one_package_raw_date(allFileDataBuff, onePackageData, readFileQueue)
        allFileDataBuff = allFileDataBuff[cutSize:]
        # 解决最后一包数据可能重复的问题
        if temptemp == onePackageData:
            debugInfo('temptemp == self.onePackageData')
            continue
        temptemp = onePackageData
        debugInfo('get onePackageData')
        # 读取每个包的识别码、CRC32、时间戳、包大小
        sOnePackageHeader = struct.Struct('8sIIIIIII')
        rawDataindex = sOnePackageHeader.size
        try:
            (recString, crc32, itermStartTimeStamp, itermEndTimeStamp, rawDataSizeAcc, rawDataSizeGyro,
                rawDataSizeTemper, rawDataSizeHeart) = sOnePackageHeader.unpack(onePackageData[0:rawDataindex])
            debugInfo(' recString: '+str(recString)+' crc32:' + hex(crc32) +
                      ' onePackageData size:'+str(len(onePackageData)))
        except:
            debugInfo('onePackageData is empty!!!')
            continue
        # 如果不是包的开头、校验不通过则寻找下一个头
        sAcc = struct.Struct('hhh')
        sGyro = struct.Struct('hhh')
        sTemper = struct.Struct('hh')
        sHeart = struct.Struct('hh')
        # acc
        accByteSize = rawDataSizeAcc*sAcc.size
        debugInfo('\nrawDataSizeAcc:'+str(rawDataSizeAcc))
        # gyro
        GyroByteSize = rawDataSizeGyro*sGyro.size
        debugInfo('\nrawDataSizeGyro:'+str(rawDataSizeGyro))
        # temper
        TemperByteSize = rawDataSizeTemper*sTemper.size
        debugInfo('\nrawDataSizeTemper:'+str(rawDataSizeTemper))
        # heart
        heartByteSize = rawDataSizeHeart*sHeart.size
        debugInfo('\nrawDataSizeHeart:'+str(rawDataSizeHeart))

        # 检查CRC32
        calcCrc32 = binascii.crc32(
            onePackageData[len(recString)+4:])
        debugInfo('calcCrc32:'+hex(calcCrc32))
        if calcCrc32 != crc32:
            debugInfo('calcCrc32 != crc32')
            continue
        # 截取原始值数据
        onePackageData = onePackageData[sOnePackageHeader.size:]
        # 解析iterm
        saveFileOnePackge = []  # 创建的是多行三列的二维列表
        maxCount = max(rawDataSizeAcc, rawDataSizeGyro,
                       rawDataSizeTemper, rawDataSizeHeart)
        if maxCount <= 0:
            continue

        # 预处理秒误差
        if itermStartTimeStamp - tempTimesStamp >= 1 and tempTimesStamp != 0:
            itermStartTimeStamp -= 1
            # print('itermStartTimeStamp - tempTimesStamp >= 1')
        tempTimesStamp = itermEndTimeStamp

        timeAmongStep = (
            (itermEndTimeStamp - itermStartTimeStamp)*1000)/(maxCount)

        debugInfo('maxCount:'+str(maxCount))
        debugInfo('itermStartTimeStamp:'+str(itermStartTimeStamp))
        debugInfo('itermEndTimeStamp:'+str(itermEndTimeStamp))
        debugInfo('time among:'+str(timeAmongStep))
        # 第一行插入remarks
        if j == 0:
            csvFileHeadDict['remarks'] = remarkesString
            csvFileHeadDict = deepcopy(csvFileHeadDict)
        for i in range(maxCount):
            timeCalc = itermStartTimeStamp*1000+i*timeAmongStep
            # timeArray = time.localtime(timeCalc/1000)
            # styleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            # csvFileHeadDict['dateTime'] = styleTime + '.' + str(int(timeCalc % 1000))
            csvFileHeadDict['dateTime'] = int(timeCalc)
            # csvFileHeadDict['dateTime'] = int(
            #     itermStartTimeStamp*1000+i*timeAmongStep)
            saveFileOnePackge.append(csvFileHeadDict)
            # 需要深度复制 否则下次更改时会连之前的列表中的值一起更改
            csvFileHeadDict = deepcopy(csvFileHeadDict)
            csvFileHeadDict['remarks'] = ''
        # 计算每个传感器的存储比例关系
        # maxCount -= 1
        temp = rawDataSizeAcc
        if temp > 0:
            accScale = maxCount/temp
        else:
            accScale = 0
        temp = rawDataSizeGyro
        if temp > 0:
            gyroScale = maxCount/temp
        else:
            gyroScale = 0
        temp = rawDataSizeTemper
        if temp > 0:
            temperScale = maxCount/temp
        else:
            temperScale = 0
        temp = rawDataSizeHeart
        if temp > 0:
            heartScale = maxCount/temp
        else:
            heartScale = 0
        debugInfo(' accScale:'+str(accScale)+' gyroScale:'+str(gyroScale) +
                  ' temperScale:'+str(temperScale)+' heartScale:'+str(heartScale))
        startTimeArray = time.localtime(itermStartTimeStamp)
        endTimeArray = time.localtime(itermEndTimeStamp)
        styleTime = time.strftime("%Y--%m--%d %H:%M:%S\n", startTimeArray)
        styleTime += time.strftime("%Y--%m--%d %H:%M:%S ", endTimeArray)
        debugInfo(str(j)+' iterm')
        debugInfo('\nstyleTime:\n'+styleTime)
        # 解析原始数据
        # Acc
        sGetRawData = struct.Struct(str(
            accByteSize)+'s'+str(GyroByteSize)+'s'+str(TemperByteSize)+'s'+str(heartByteSize)+'s')
        try:
            (accRawData, gyroRawData, temperRawData,
                heartRawData) = sGetRawData.unpack(onePackageData[0:])
        except:
            continue
        for i in range(0, rawDataSizeAcc):
            accIndex = int(i*accScale)
            try:
                (saveFileOnePackge[accIndex]['acc_x'], saveFileOnePackge[accIndex]['acc_y'],
                    saveFileOnePackge[accIndex]['acc_z']) = sAcc.unpack(accRawData[0:sAcc.size])
            except:
                continue
            saveFileOnePackge[accIndex]['acc_x'] = calcAccGryro(
                saveFileOnePackge[accIndex]['acc_x'], accRange)
            saveFileOnePackge[accIndex]['acc_y'] = calcAccGryro(
                saveFileOnePackge[accIndex]['acc_y'], accRange)
            saveFileOnePackge[accIndex]['acc_z'] = calcAccGryro(
                saveFileOnePackge[accIndex]['acc_z'], accRange)
            accRawData = accRawData[sAcc.size:]
        # Gyro
        for i in range(0, rawDataSizeGyro):
            gyroIndex = int(i*gyroScale)
            try:
                (saveFileOnePackge[gyroIndex]['gyr_x'], saveFileOnePackge[gyroIndex]['gyr_y'],
                    saveFileOnePackge[gyroIndex]['gyr_z']) = sGyro.unpack(gyroRawData[0:sGyro.size])
            except:
                continue
            saveFileOnePackge[gyroIndex]['gyr_x'] = calcAccGryro(
                saveFileOnePackge[gyroIndex]['gyr_x'], gyroRange)
            saveFileOnePackge[gyroIndex]['gyr_y'] = calcAccGryro(
                saveFileOnePackge[gyroIndex]['gyr_y'], gyroRange)
            saveFileOnePackge[gyroIndex]['gyr_z'] = calcAccGryro(
                saveFileOnePackge[gyroIndex]['gyr_z'], gyroRange)
            gyroRawData = gyroRawData[sGyro.size:]
        # temperate
        for i in range(0, rawDataSizeTemper):
            temperIndex = int(i*temperScale)
            try:
                (saveFileOnePackge[temperIndex]['bodySurface_temp'], saveFileOnePackge[temperIndex]
                    ['ambient_temp']) = sTemper.unpack(temperRawData[0:sTemper.size])
            except:
                continue
            saveFileOnePackge[temperIndex]['bodySurface_temp'] = saveFileOnePackge[temperIndex]['bodySurface_temp']/10
            saveFileOnePackge[temperIndex]['ambient_temp'] = saveFileOnePackge[temperIndex]['ambient_temp']/10
            temperRawData = temperRawData[sTemper.size:]
        # heartrate
        for i in range(0, rawDataSizeHeart):
            heartIndex = int(i*heartScale)
            try:
                (saveFileOnePackge[heartIndex]['hr_raw'], saveFileOnePackge[heartIndex]
                    ['hr']) = sHeart.unpack(heartRawData[0:sHeart.size])
            except:
                continue
            heartRawData = heartRawData[sHeart.size:]

        csv_write_dict(
            csv_file, csvFileHead[0], saveFileOnePackge)
        percentCount += 1
        percent = int((percentCount / headerPackeNum)*100)
        if lastPercent != percent:
            lastPercent = percent
        flashSting = ['-', '\\', '|', '/']
        print(flashSting[percentCount % len(flashSting)] +
              ' '+str(lastPercent)+'%', end='\r', flush=True)
    return 0


def ReadFileThread(bin_file, readFileQueue):  # 建立一个任务线程类
    readOpenFile = None
    fileQueue = readFileQueue
    readFile = bin_file
    try:
        readOpenFile = open(readFile, 'rb')
    except:
        debugInfo('can not open file')
        return
    while True:
        debugInfo('put one iterm to queue')
        data = readOpenFile.read(QUEUE_ONE_ITERM_BYTE_SIZE)
        fileQueue.put(data)
        if len(data) != QUEUE_ONE_ITERM_BYTE_SIZE:
            break
    readOpenFile.close()
    debugInfo('all file data put to queue')


# bin2csv('D:\\MATA00-1000777-20210603-104952.BIN', '.\\test12.csv')