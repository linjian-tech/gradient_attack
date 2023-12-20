##########################################################
#Udata and Sdata are the output values
##########################################################
import numpy as np
import time
import random
from numpy import linalg as LA
import torch
import copy
import math

def local_update(single_user_vector, rateList, user_rating_list, item_vector,  para):
    eta = para['eta']
    reg_u = para['lambda']
    gradient = torch.zeros([len(item_vector), len(single_user_vector)])
    temp_item_vector = copy.deepcopy(item_vector)
    t_temp_item_vector = copy.deepcopy(temp_item_vector)
    Iters = para['local_iter']
    for iter in range(Iters):
        random.shuffle(user_rating_list)
        t_temp_item_vector = temp_item_vector * 1
        for item_id in user_rating_list:
            error = torch.dot(single_user_vector, t_temp_item_vector[item_id]) - rateList[item_id]#
            single_user_vector = single_user_vector - eta * (error * t_temp_item_vector[item_id] + reg_u * single_user_vector)#
            temp_item_vector[item_id] =  temp_item_vector[item_id] - eta *( error * single_user_vector + reg_u * temp_item_vector[item_id])
            gradient[item_id] = error * single_user_vector + reg_u * temp_item_vector[item_id]
    return single_user_vector, (item_vector - temp_item_vector)/eta



def collect(train_matrix, num_user, num_service, para):
    # initialize
    dimention = para['dimension']
    lmda = para['lambda']
    eta = para['eta']
    maxIter = para['reconstruct_max_iter']
    metrics = para['metrics']
    continue_round = para['continue_round']
    user_vector = torch.randn(num_user, dimention)
    service_vector = torch.randn(num_service, dimention)

    # collect index of user matrix
    user_matrix_index = []
    for i in range(num_user):
        user_matrix_index.append([])
        for j in range(num_service):
            if(train_matrix[i][j] > 0):
                user_matrix_index[i].append(j)

    # select attacked user, in this code, we select user 0
    attacked_user = 0
    selected_user_matrix_index = copy.deepcopy(user_matrix_index[attacked_user])
    selected_user_vector = copy.deepcopy(user_vector[attacked_user])

    # attacker begin to collect user gradient and public service user vector
    gradient_collection = []
    service_vector_collection = []
    for _ in range(continue_round):
        service_vector_collection.append(copy.deepcopy(service_vector))

        gradient_from_user = []
        for i in range(num_user):
            user_vector[i], gradient = local_update(user_vector[i], train_matrix[i], selected_user_matrix_index, service_vector, para)
            gradient_from_user.append(gradient)
        # server update global model
        for g in gradient_from_user:
            service_vector -= eta * g

        gradient_collection.append(copy.deepcopy(gradient_from_user[attacked_user]))
    return train_matrix[attacked_user], selected_user_vector, user_matrix_index[attacked_user], service_vector_collection, gradient_collection


def reconstruct(num_user, num_service, gradient_collection, service_vector_collection, dummy_user_vector, dummy_QoS, selected_user_matrix_index, para):
    max_iter = para['reconstruct_max_iter']
    threshold_loss = para['thresholds']
    continue_round = para['continue_round']

    origin_user_matrix_index = copy.deepcopy(selected_user_matrix_index)
    reconstruct_user_vector, reconstruct_QoS = dummy_user_vector.clone().detach(), dummy_QoS[origin_user_matrix_index].clone().detach()
    min_loss = 100000
    min_iter = 0
    optimizer = torch.optim.LBFGS([dummy_user_vector, dummy_QoS])
    for iter in range(max_iter):
        temp_dummy_user_vector = [[] for _ in range(continue_round+1)]
        temp_dy_dx = [[] for _ in range(continue_round)]
        temp_dummy_user_vector[0] = dummy_user_vector
        def closure():
            optimizer.zero_grad()
            for _ in range(continue_round):
                temp_dummy_user_vector[_+1], temp_dy_dx[_] = local_update(temp_dummy_user_vector[_], dummy_QoS, selected_user_matrix_index, service_vector_collection[_] , para)

            distance = 0
            for _ in range(continue_round):
                for gx, gy in zip(temp_dy_dx[_][origin_user_matrix_index], gradient_collection[_][origin_user_matrix_index]):
                    distance += ((gx - gy) ** 2).sum()
            distance.backward()
            return distance


        optimizer.step(closure)
        current_loss = closure()

        # averageing loss
        loss = current_loss.item()/continue_round/len(selected_user_matrix_index)

        if math.isnan(loss):
            min_loss = loss
            print("this rescontruction is fail")
            break
        if loss < min_loss:
            min_loss = loss
            min_iter = iter
            reconstruct_user_vector, reconstruct_QoS = dummy_user_vector.clone().detach(), dummy_QoS[origin_user_matrix_index].clone().detach()
        print(iter, "iter: loss = %.4f " % (loss))
        if min_loss <threshold_loss:
            break
    return min_iter+1, min_loss, reconstruct_user_vector, reconstruct_QoS





#======================================================#
# Function to remove the entries of data matrix
# Return the trainMatrix and testMatrix
#======================================================#

def removeEntries(matrix, density, seedId):
    (vecX, vecY) = np.where(matrix > 0)
    vecXY = np.c_[vecX, vecY]
    numRecords = vecX.size
    numAll = matrix.size
    random.seed(seedId)
    randomSequence = list(range(0, numRecords))
    random.shuffle(randomSequence)
    numTrain = int(numAll * density)
    # by default, we set the remaining QoS records as testing data
    numTest = numRecords - numTrain
    trainXY = vecXY[randomSequence[0 : numTrain], :]
    testXY = vecXY[randomSequence[- numTest :], :]

    trainMatrix = np.zeros(matrix.shape)
    trainMatrix[trainXY[:, 0], trainXY[:, 1]] = matrix[trainXY[:, 0], trainXY[:, 1]]
    testMatrix = np.zeros(matrix.shape)
    testMatrix[testXY[:, 0], testXY[:, 1]] = matrix[testXY[:, 0], testXY[:, 1]]

    # ignore invalid testing data
    idxX = (np.sum(trainMatrix, axis=1) == 0)
    testMatrix[idxX, :] = 0
    idxY = (np.sum(trainMatrix, axis=0) == 0)
    testMatrix[:, idxY] = 0
    return trainMatrix, testMatrix


#======================================================#
# Function to compute the evaluation metrics
#======================================================#
def errMetric(realVec, predVec, metrics):
    result = []
    absError = np.absolute(predVec - realVec)
    mae = np.average(absError)
    for metric in metrics:
        if 'MAE' == metric:
            result = np.append(result, mae)
        if 'NMAE' == metric:
            nmae = mae / np.average(realVec)
            result = np.append(result, nmae)
        if 'RMSE' == metric:
            rmse = LA.norm(absError) / np.sqrt(absError.size)
            result = np.append(result, rmse)

    return result


def executeOneSetting(matrix, density, para):
    #initialize
    dimension = para['dimension']
    continue_round = para['continue_round']
    count = para['repeat_experiment']
    random_seed = 1
    (num_user, num_service) = matrix.shape

    # load the training data
    (train_matrix, test_matrix) = removeEntries(matrix, para['density'], random_seed)

    # attacker: collect user service vector gradients and service vector
    (selected_user_matrix, selected_user_vector, selected_user_matrix_index, service_vector_collection, gradient_collection) = collect(train_matrix, num_user, num_service, para)
    origin_user_matrix_index = copy.deepcopy(selected_user_matrix_index)
    # attacker: execute attack
    dummy_user_vector = torch.randn(dimension).requires_grad_(True)
    dummy_QoS = torch.randn(num_service).requires_grad_(True)
    min_iter, loss, dummy_user_vector, dummy_QoS = reconstruct(num_user, num_service, gradient_collection, service_vector_collection,\
                                                               dummy_user_vector, dummy_QoS, selected_user_matrix_index, para)
    # bias correction
    selected_user_vector = selected_user_vector.detach().numpy()
    dummy_user_vector = dummy_user_vector.detach().numpy()
    dummy_QoS = dummy_QoS.detach().numpy()
    if np.sum(dummy_QoS) < 0:
        dummy_user_vector = - dummy_user_vector
        dummy_QoS = - dummy_QoS
    dummy_QoS = np.maximum(dummy_QoS,0)
    # compute reconstruct error
    rmse_QoS = errMetric(selected_user_matrix[origin_user_matrix_index], dummy_QoS,['RMSE'])
    rmse_user_vector = errMetric(selected_user_vector, dummy_user_vector,['RMSE'])
    print("rmse_QoS:{}, rmse_user_vector:{}".format(rmse_QoS, rmse_user_vector))
    print("raw_data:{}".format(selected_user_matrix[origin_user_matrix_index]))
    print("Reconstruct data:{}".format(dummy_QoS))


if __name__ == '__main__':
    # parameter setting
    para = {'dataPath': 'dataset/rtMatrix.txt',
            'outPath': 'result/',
            'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NPRE'], # evaluate matric
            'density': 0.001, # dataset density
            'dimension': 10, # dimenisionality of the latent vectors
            'eta': 0.001, # learning rate
            'lambda': 0.01, # regularization parameter
            'reconstruct_max_iter': 100, # the max iterations for reconstruction
            'local_iter': 5,
            'thresholds': 0.1, # if loss < thresholds, then the reconstruction stops
            'continue_round': 3, # how many continue rounds gradients are collected
            'repeat_experiment':10, # how many runs are performed at each matrix density
            }
    # load dataset
    dataset = np.loadtxt(para['dataPath'])

    executeOneSetting(dataset, para['density'],para)

