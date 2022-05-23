import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from numpy import argmax
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score,f1_score, roc_curve, precision_recall_curve


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('grid', linestyle="dotted", color='black')

def train(epochs, model, trainloader, testloader, optimizer, args, log):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            #loss = nn.NLLLoss(output,target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.batch_size == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(trainloader.sampler)
                print(f'Train Epoch: {epoch} Loss: {loss.item():.6f}')
                log.info("Train Epoch: {} Loss: {}".format(epoch,loss.item()))

        test(model, testloader, args, log)
    return model


def test(model, loader, args, log):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(loader.sampler)
        accuracy = 100. * correct / len(loader.sampler)
        log.info("Accuracy: {}/{} ({}%)".format(correct,len(loader.sampler),accuracy))
    return accuracy


def attinfattack(X_adv_train, Z_adv_train, X_adv_test, Z_adv_test, args, log):

    if args.dataset == "LAW":
        attack_model_race = RandomForestClassifier(max_depth=150, random_state=1337)
    else:
        attack_model_race = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(64,128,32,), verbose=0, max_iter=500,random_state=1337)
    attack_model_input = pd.DataFrame(X_adv_train)
    attack_model_race.fit(X_adv_train, Z_adv_train['race'])
    z_pred_race_prob = attack_model_race.predict_proba(attack_model_input)
    z_pred_race_prob = z_pred_race_prob[:, 1]

    if args.dataset == "LAW":
        attack_model_sex = RandomForestClassifier(max_depth=150, random_state=1337)
    else:
        attack_model_sex = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(64,128,32,), verbose=0, max_iter=500,random_state=1337)
    attack_model_sex.fit(X_adv_train, Z_adv_train['sex'])
    z_pred_sex_prob = attack_model_sex.predict_proba(attack_model_input)
    z_pred_sex_prob = z_pred_sex_prob[:, 1]

    # get the best threshold for Precision Recall Curve: RACE
    precision_race, recall_race, thresholds_race_pr = precision_recall_curve(Z_adv_train['race'], z_pred_race_prob)
    fscore_race = (2 * precision_race * recall_race) / (precision_race + recall_race)
    if np.isnan(fscore_race).any():
        # best_thresh_race = 0.5
        best_thresh_race = np.nanargmax(fscore_race)
    else:
        best_thresh_race = thresholds_race_pr[argmax(fscore_race)]
        best_fscore_race = fscore_race[argmax(fscore_race)]
    log.info('Best Threshold for RACE attribute (Precision Recall)= {}, F-Score= {}'.format(best_thresh_race, best_fscore_race))
    print("Best Threshold (RACE) {}".format(best_thresh_race))

    # get the best threshold for Precision Recall Curve: SEX
    precision_sex, recall_sex, thresholds_sex_pr = precision_recall_curve(Z_adv_train['sex'], z_pred_sex_prob)
    fscore_sex = (2 * precision_sex * recall_sex) / (precision_sex + recall_sex)
    if np.isnan(fscore_sex).any():
        # best_thresh_sex = 0.5
        best_thresh_sex = np.nanargmax(fscore_sex)
    else:
        best_thresh_sex = thresholds_sex_pr[argmax(fscore_sex)]
        best_score_sex = fscore_sex[argmax(fscore_sex)]
    log.info('Best Threshold for SEX attribute (Precision Recall)= {}, F-Score= {}'.format(best_thresh_sex, best_score_sex))
    print("Best Threshold (Sex) {}".format(best_thresh_sex))

    # thresholding on test dataset
    attack_model_input = pd.DataFrame(X_adv_test)
    z_pred_sex_prob = attack_model_sex.predict_proba(attack_model_input)
    z_pred_sex_prob = z_pred_sex_prob[:, 1]
    z_pred_race_prob = attack_model_race.predict_proba(attack_model_input)
    z_pred_race_prob = z_pred_race_prob[:, 1]

    z_pred_race = z_pred_race_prob > best_thresh_race
    err_race = np.abs(z_pred_race_prob - best_thresh_race)
    z_pred_sex = z_pred_sex_prob > best_thresh_sex
    err_sex = np.abs(z_pred_sex_prob - best_thresh_sex)

    log.info("####### Attack Success on Race Attribute #######")
    log.info("Recall: {}".format(recall_score(Z_adv_test['race'], z_pred_race)))
    log.info("Precision: {}".format(precision_score(Z_adv_test['race'], z_pred_race)))
    log.info("F1 Score: {}".format(f1_score(Z_adv_test['race'], z_pred_race)))

    log.info("####### Attack Success on Gender Attribute #######")
    log.info("Recall: {}".format(recall_score(Z_adv_test['sex'], z_pred_sex)))
    log.info("Precision: {}".format(precision_score(Z_adv_test['sex'], z_pred_sex)))
    log.info("F1 Score: {}".format(f1_score(Z_adv_test['sex'], z_pred_sex)))

    random_guess = len(Z_adv_test['sex'][Z_adv_test['sex']==1]) / len(Z_adv_test['sex'])
    fig = plt.figure(figsize=(4.5, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0,1], [random_guess,random_guess], linestyle='--', label='Random Guess')
    ax.plot(recall_sex, precision_sex, color='0.50', ls='solid', marker='.', markersize=4, label="Gender")
    ax.plot(recall_race, precision_race, color='black', ls='solid', marker='.', markersize=4, label="Race")
    ax.set_xlabel('Recall', fontsize='large')
    ax.set_ylabel('Precision',fontsize='large')
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./fairadv_pr_{}_{}_{}_{}.pdf".format(args.dataset,args.with_sattr,args.attfeature,args.explanations), bbox_inches = 'tight',pad_inches = 0, dpi=400)



def attinf_fides_phi_s(X_adv_train_race, Z_adv_train_race, X_adv_test_race, Z_adv_test_race, X_adv_train_sex, Z_adv_train_sex, X_adv_test_sex, Z_adv_test_sex, args, log):

    if args.dataset == "LAW":
        attack_model_race = RandomForestClassifier(max_depth=150, random_state=1337)
    else:
        attack_model_race = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(64,128,32,), verbose=0, max_iter=500,random_state=1337)
    attack_model_input = pd.DataFrame(X_adv_train_race)
    attack_model_race.fit(X_adv_train_race, Z_adv_train_race)
    z_pred_race_prob = attack_model_race.predict_proba(attack_model_input)
    z_pred_race_prob = z_pred_race_prob[:, 1]

    if args.dataset == "LAW":
        attack_model_sex = RandomForestClassifier(max_depth=150, random_state=1337)
    else:
        attack_model_sex = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(64,128,32,), verbose=0, max_iter=500,random_state=1337)
    attack_model_sex.fit(X_adv_train_sex, Z_adv_train_sex)
    z_pred_sex_prob = attack_model_sex.predict_proba(attack_model_input)
    z_pred_sex_prob = z_pred_sex_prob[:, 1]


    # get the best threshold for Precision Recall Curve: RACE
    precision_race, recall_race, thresholds_race_pr = precision_recall_curve(Z_adv_train_race, z_pred_race_prob)
    fscore_race = (2 * precision_race * recall_race) / (precision_race + recall_race)
    if np.isnan(fscore_race).any():
        # best_thresh_race = 0.5
        best_thresh_race = np.nanargmax(fscore_race)
    else:
        best_thresh_race = thresholds_race_pr[argmax(fscore_race)]
        best_fscore_race = fscore_race[argmax(fscore_race)]
    log.info('Best Threshold for RACE attribute (Precision Recall)= {}, F-Score= {}, Precision {}, Recall {}'.format(best_thresh_race, best_fscore_race,precision_race[argmax(fscore_race)],recall_race[argmax(fscore_race)]))
    print("Best Threshold (RACE) {}".format(best_thresh_race))

    # get the best threshold for Precision Recall Curve: SEX
    precision_sex, recall_sex, thresholds_sex_pr = precision_recall_curve(Z_adv_train_sex, z_pred_sex_prob)
    fscore_sex = (2 * precision_sex * recall_sex) / (precision_sex + recall_sex)
    if np.isnan(fscore_sex).any():
        # best_thresh_sex = 0.5
        best_thresh_sex = np.nanargmax(fscore_sex)
    else:
        best_thresh_sex = thresholds_sex_pr[argmax(fscore_sex)]
        best_score_sex = fscore_sex[argmax(fscore_sex)]
    log.info('Best Threshold for SEX attribute (Precision Recall)= {}, F-Score= {}, Precision {}, Recall {}'.format(best_thresh_sex, best_score_sex,precision_sex[argmax(fscore_race)],recall_sex[argmax(fscore_race)]))
    print("Best Threshold (Sex) {}".format(best_thresh_sex))

    # thresholding on test dataset
    attack_model_input = pd.DataFrame(X_adv_test_sex)
    z_pred_sex_prob = attack_model_sex.predict_proba(attack_model_input)
    z_pred_sex_prob = z_pred_sex_prob[:, 1]
    attack_model_input = pd.DataFrame(X_adv_test_race)
    z_pred_race_prob = attack_model_race.predict_proba(attack_model_input)
    z_pred_race_prob = z_pred_race_prob[:, 1]

    z_pred_race = z_pred_race_prob > best_thresh_race
    z_pred_sex = z_pred_sex_prob > best_thresh_sex

    log.info("####### Attack Success on Race Attribute #######")
    log.info("Recall: {}".format(recall_score(Z_adv_test_race, z_pred_race)))
    log.info("Precision: {}".format(precision_score(Z_adv_test_race, z_pred_race)))
    log.info("F1 Score: {}".format(f1_score(Z_adv_test_race, z_pred_race)))

    log.info("####### Attack Success on Gender Attribute #######")
    log.info("Recall: {}".format(recall_score(Z_adv_test_sex, z_pred_sex)))
    log.info("Precision: {}".format(precision_score(Z_adv_test_sex, z_pred_sex)))
    log.info("F1 Score: {}".format(f1_score(Z_adv_test_sex, z_pred_sex)))


def prediction_adversary(clf, X_adv_train, y_adv_train, Z_adv_train, X_adv_test, y_adv_test, Z_adv_test, args, log):

    log.info("####### Attribute Inference Attack with only output predictions #######")

    y_prob = clf.predict_proba(X_adv_train)
    y_prob = pd.DataFrame(y_prob, columns = ['class1','class2'])
    attack_model_input = pd.DataFrame(clf.predict_proba(X_adv_train), columns = ['class1','class2'])


    attack_model_race = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(64,128,32,), verbose=0, max_iter=300,random_state=1337)
    attack_model_race.fit(y_prob, Z_adv_train['race'])
    z_pred_race_prob = attack_model_race.predict_proba(attack_model_input)
    z_pred_race_prob = z_pred_race_prob[:, 1]


    attack_model_sex = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(64,128,32,), verbose=0, max_iter=300,random_state=1337)
    attack_model_sex.fit(y_prob, Z_adv_train['sex'])
    z_pred_sex_prob = attack_model_sex.predict_proba(attack_model_input)
    z_pred_sex_prob = z_pred_sex_prob[:, 1]


    # get the best threshold for Precision Recall Curve: RACE
    precision_race, recall_race, thresholds_race_pr = precision_recall_curve(Z_adv_train['race'], z_pred_race_prob)
    fscore_race = (2 * precision_race * recall_race) / (precision_race + recall_race)
    best_thresh_race = thresholds_race_pr[argmax(fscore_race)]
    best_fscore_race = fscore_race[argmax(fscore_race)]
    log.info('Best Threshold for RACE attribute (Precision Recall)= {}, F-Score= {}'.format(best_thresh_race, best_fscore_race))

    # get the best threshold for Precision Recall Curve: SEX
    precision_sex, recall_sex, thresholds_sex_pr = precision_recall_curve(Z_adv_train['sex'], z_pred_sex_prob)
    fscore_sex = (2 * precision_sex * recall_sex) / (precision_sex + recall_sex)
    best_thresh_sex = thresholds_sex_pr[argmax(fscore_sex)]
    best_score_sex = fscore_sex[argmax(fscore_sex)]
    log.info('Best Threshold for SEX attribute (Precision Recall)= {}, F-Score= {}'.format(best_thresh_sex, best_score_sex))

    # thresholding on test dataset
    attack_model_input = pd.DataFrame(clf.predict_proba(X_adv_test), columns = ['class1','class2'])
    z_pred_sex_prob = attack_model_sex.predict_proba(attack_model_input)
    z_pred_sex_prob = z_pred_sex_prob[:, 1]
    z_pred_race_prob = attack_model_race.predict_proba(attack_model_input)
    z_pred_race_prob = z_pred_race_prob[:, 1]

    z_pred_race = z_pred_race_prob > best_thresh_race
    z_pred_sex = z_pred_sex_prob > best_thresh_sex

    log.info("####### Attack Success on Race Attribute #######")
    log.info("Recall: {}".format(recall_score(Z_adv_test['race'], z_pred_race)))
    log.info("Precision: {}".format(precision_score(Z_adv_test['race'], z_pred_race)))
    log.info("F1 Score: {}".format(f1_score(Z_adv_test['race'], z_pred_race)))

    log.info("####### Attack Success on Gender Attribute #######")
    log.info("Recall: {}".format(recall_score(Z_adv_test['sex'], z_pred_sex)))
    log.info("Precision: {}".format(precision_score(Z_adv_test['sex'], z_pred_sex)))
    log.info("F1 Score: {}".format(f1_score(Z_adv_test['sex'], z_pred_sex)))


    random_guess = len(Z_adv_test['sex'][Z_adv_test['sex']==1]) / len(Z_adv_test['sex'])
    fig = plt.figure(figsize=(4.5, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0,1], [random_guess,random_guess], linestyle='--', label='Random Guess')
    ax.plot(recall_sex, precision_sex, color='0.50', ls='solid', marker='.', markersize=4, label="Gender")
    if not args.dataset == "LFW":
        ax.plot(recall_race, precision_race, color='black', ls='solid', marker='.', markersize=4, label="Race")
    #ax.title.set_text('{} Precision-Recall Curve'.format(args.dataset))
    ax.set_xlabel('Recall', fontsize='large')
    ax.set_ylabel('Precision',fontsize='large')
    if args.dataset == "LAW" or "CREDIT":
        plt.legend(loc="center right")
    else:
        plt.legend(loc="lower left")
    plt.grid()
    plt.savefig("./fairadv_pr_{}.pdf".format(args.dataset), bbox_inches = 'tight',pad_inches = 0, dpi=400)
