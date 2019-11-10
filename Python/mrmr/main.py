import os as os
import numpy as np
import sklearn




def readData(fileName, relativePath = "", skip_header=0, d=','):
    '''
    retrieve the data from the file named fileName
    you may have to modify the path below in other OSs.
    in Linux, Mac OS, the partition is "/". In Windows OS, the partition is "\" (backslash).
    os.sep means "\" in windows. So the following path is adjusted for Windows OS

    param filename: filename
    param relativePath: the relative path from "data" folder to the folder where there is a file you want to import
    param d: the type of data separator, which s "," by default
    return: a numpy.array of the data
    '''
    # read a file in "data" folder
    if relativePath == "":
        filePath = os.path.dirname(__file__).replace('/', os.sep) + '\\' + 'data\\' + fileName
        print("filePath:{}".format(filePath))
        print("os.path.dirname(__file__):{}".format(os.path.dirname(__file__)))

    else:
        filePath = os.path.dirname(__file__).replace('/', os.sep) + '\\' + 'data\\' + relativePath + '\\' + fileName
        # print "filePath:{}".format(filePath)

    if skip_header == 0:
        return np.genfromtxt(filePath, delimiter=d, dtype=None)
    else:
        return np.genfromtxt(filePath, delimiter=d, dtype=None, skip_header=skip_header)



# import the electricity sales price file: source (download the CSV file): https://www.eia.gov/electricity/data/browser/#/topic/7?agg=0,1&geo=0000000001&endsec=vg&freq=M&start=200101&end=201802&ctype=linechart&ltype=pin&rtype=s&maptype=0&rse=0&pin=
fileName = constant.averageRetailPriceOfElectricityMonthly
# import the file removing the header
fileData = Util.readData(fileName, relativePath="", skip_header=1, d='\t')
# print ("fileData:{}".format(fileData))


# TODO
# import dataset




