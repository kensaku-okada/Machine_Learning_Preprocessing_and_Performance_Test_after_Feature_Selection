#!/usr/bin/python3
#coding:utf-8
import os, sys, csv, pathlib, glob
import arff # for sparse arff dataset
import pandas as pd
import Constant

def getFilePath(relativePath):

    # the directory where the main script is
    main_script_dir = os.getcwd()

    os.chdir("..\\..")
    print("os.path.abspath(os.curdir): ",os.path.abspath(os.curdir))

    filePath = os.path.abspath(os.curdir) + '\\' + relativePath
    # print("filePath: ",filePath)

    # get back to the the directory where the main script is
    os.chdir(main_script_dir)

    return filePath

# def readArffData(fileName, relativePath = "", skip_header=0, d=','):
def importArffData(filePath, config):
    '''
    source:
        https://discuss.analyticsvidhya.com/t/loading-arff-type-files-in-python/27419
        https://codeday.me/jp/qa/20190205/202735.html
    param relativePath: path to the file form the great parent folder
    return: dataset as pandas
    '''

    if config.outputDatasetType == "dense":
        # dataset = scipy_arff.loadarff(filePath)
        # # print("dataset[1]: ", dataset[1])
        # df = pd.DataFrame(dataset[0])

        data = arff.load(open(filePath))
        # print("data: ", data)
        # print("type(data): ", type(data))
        # print("len(data): ", len(data))

    elif config.outputDatasetType == "sparsed":
        # https://pypi.org/project/liac-arff/
        # https://laats.github.io/sw/mit/arff/
        # http://scikit.ml/datasets.html
        # (name, sparse, alist, m) = arff.arffread(open(filePath))

        # https://pythonhosted.org/liac-arff/
        data = arff.load(open(filePath), return_type=arff.LOD)

        # if you want to CV by sparse data, refer to the following:
        # https://stackoverflow.com/questions/33588658/stratified-kfold-on-sparsecsr-feature-matrixprint("data:", data)
        # https://medium.com/@yamasaKit/3-sparse%E3%81%AA%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AB%E5%AF%BE%E3%81%99%E3%82%8Bscikit-learn%E3%81%AEsvm%E3%82%92svmlight-file%E3%81%A7%E9%AB%98%E9%80%9F%E5%8C%96%E3%81%99%E3%82%8B-4cc9e13529e9

    else:
        print("undefied output data type specified. stop the program")
        sys.exit(0)


    df = pd.DataFrame(data['data'])
    return df

def ifExportFileExists(config, filePath):

    file_name = pathlib.Path(filePath).stem

    currentDir = os.getcwd()
    # print("currentDir at ifExportFileExists: ", currentDir)

    exportFilePath = getExportFileDir(config, currentDir) + "\\" + getExportFileName(file_name)
    # print("exportFilePath at ifExportFileExists: ", exportFilePath)
    # print("type(exportFilePath) at ifExportFileExists: ", type(exportFilePath))

    # existingExportFilePaths = list(pathlib.Path(getExportFileDir(config, currentDir)).glob('*.csv'))
    existingExportFilePaths = glob.glob(getExportFileDir(config, currentDir) + "\\*.csv")
    # print("existingExportFilePaths at ifExportFileExists: ", existingExportFilePaths)

    if exportFilePath in existingExportFilePaths:
        print("the export file " + getExportFileName(file_name) + " already exists. skip the iteration.")
        return True
    else:
        return False

def getExportFileName(file_name):
    return file_name + "_test_result.csv"

def getExportFileDir(config, currentDir):
    feature_selection_algorithm_name = config.feature_selection_algorithm_name
    return currentDir + "\\" + Constant.FILE_EXPORT_PATH + "\\" + feature_selection_algorithm_name

# export dataset
def exportCSVFile(config, header, filePath):

    file_name = pathlib.Path(filePath).stem

    dataSet = config.test_results

    currentDir = os.getcwd()
    print("currentDir: ", currentDir)
    #############################################
    # change the directory to export
    os.chdir(getExportFileDir(config, currentDir))
    #############################################

    # #############################################
    # # depending on the change of design, you may want to save the outputs to the following path
    # os.chdir("\\\\192.168.1.60\\eGIS\\Development\\SecurityDoctor\\report\\output")
    # #############################################

    f = open(getExportFileName(file_name) + ".csv", 'w', encoding='utf-8')  # open the file with writing mode
    csvWriter = csv.writer(f, lineterminator="\n")
    # print('header',header)
    # print('dataSet[0]',dataSet[0])

    # is header is defined
    if header is not None:
        # write the header
        csvWriter.writerow(header)

    # write each row
    for data in dataSet:
        csvWriter.writerow(data)
    f.close()  # close the file

    # take back the current directory
    os.chdir(currentDir)
    # print "os.getcwd():{}".format(os.getcwd())
