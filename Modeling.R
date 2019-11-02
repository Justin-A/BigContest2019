###########################
### 잔존 여부 예측 모델 ###
###########################

# assign label
fact <- ifelse(target1 > 60,1,0)
# data split, dt : train_data(70% on survival time)
dt <- train[ss,]
fs <- fact[ss]

# for lightGBM form
dtrain <- lgb.Dataset(rbind(dt),label=c(fs))
dvalid <- lgb.Dataset(cbind(dt),label=fs)
valids <- list(test = dvalid)
head(train)

# modeling
params <-  list(objective = "binary", metric = "AUC", device = "gpu")
model2 <- lgb.train(params, dtrain, 2000, valids,  eval_freq = 500,
                    learning_rate = 0.05, early_stopping_rounds = 100)

# validation data
vd <- train[-ss,]

# p : predict value, predict proba, 0~1사이의 값
p <- predict(model2, vd)
# p3 : predict proba to survival time, 0~1사이의 값에 64를 곱함으로써 survival form으로 변환
p3<-p*64
# over 63 to 64, under 1 to 1
pe3<-ifelse(p3>63,64,p3)
pe3<-ifelse(p3<1,1,p3)

# binary output, threshold = 0.5
predicted <- ifelse((p)/1 > 0.5,1,0)

# validation label
vali_y <- fact[-ss]

# Check True Negative : 0.7058
sum(predicted==0 & vali_y==0)/sum(vali_y==0)
# Check True Positive : 0.8261
sum(predicted==1 & vali_y==1)/sum(vali_y==1)
# Acc : 0.7730
acc1 <- sum(predicted==vali_y)/length(vali_y) ## acc1

###############################
#### 모델 보정을 위한 모델 ####
###############################
## => 대부분 예측값이 1로 쏠림, 해당 코드의 목적은? 확실하게 돈을 아예 사용하지 않은 acc_id 예측으로 추정

# *target1 : survival time / target2 : amount spent
# new_target2 : survival_time과 amount_spent의 곱으로 총 누적 결제금액을 뜻함
new_target2<-target1*target2
# new_target2 summary
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.0000  0.0000  0.9744  3.9249  4.6066 97.5155

# if amount spent == 0 -> new_target3 == 0
# new_target3 : amount_spent가 0이면 new_target3도 0, 그외는 1
new_target3<-(ifelse(new_target2>0,1,0))

# for lightGBM form
dtrain <- lgb.Dataset(rbind(train[ss,]),label=c(new_target3[ss]))
dvalid <- lgb.Dataset(rbind(train[ss,]),label=new_target3[ss])
valids <- list(test = dvalid)

# modeling
params <- list(objective = "binary", metric = "AUC", device = "gpu")
model3 <- lgb.train(params, dtrain, 2000, valids,  eval_freq = 500,
                    learning_rate = 0.05, early_stopping_rounds = 100)

# predict value
# p : predict value, predict proba, 0~1사이의 값
p <- predict(model3, train[-ss,])
ppay <- p
# p summary
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# 0.0000684 0.0840116 0.8587072 0.6094596 0.9924873 0.9999988 

# why divide by 1?
predicted_p <- ifelse((p)/1 > 0.01,1,0)
# predicted_p summary
# Min.         1st Qu.          Median            Mean         3rd Qu.            Max. 
# 0.0000000000000 1.0000000000000 1.0000000000000 0.8903333333333 1.0000000000000 1.0000000000000 
# validation label
vali_y<-new_target3[-ss]

# Check True Negative : 0.077, 0.297
sum(predicted_p==0 & vali_y==0)/sum(vali_y==0)
# 0.7156, 0.99
sum(predicted_p==0 & vali_y==0)/sum(predicted_p==0)
# Check True Positive : 0.9786, 0.99
sum(predicted_p==1 & vali_y==1)/sum(vali_y==1)

# Acc : 0.6085, 0.70
acc_pay <- sum(predicted_p==vali_y)/length(vali_y)

###############################################
#### 총 누적결제 금액을 예측하기 위한 모델 ####
###############################################

# new_target2 summary
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.0000  0.0000  0.9744  3.9249  4.6066 97.5155
new_target2 <- target1*target2
tts <- target2
# train label
nt <- new_target2[ss]

# for lightGBM form
dtrain <- lgb.Dataset(rbind(train[ss,]),label=c(new_target2[ss]))
dvalid <- lgb.Dataset(rbind(train[ss,]),label=new_target2[ss])
valids <- list(test = dvalid)

# modeling
params <- list(objective = "regression_l2", metric = "l2", device = "gpu")
model <- lgb.train(params, dtrain, 2000, valids, eval_freq = 500,
                   learning_rate = 0.02, early_stopping_rounds = 100)

# predict value
p <- predict(model, train[-ss,])
# p summary
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -1.8941  0.5339  2.5231  4.0032  5.9381 23.9932

# validation label
vali_y2 <- (new_target2[-ss])
# vali_y2 summary
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.0000  0.0000  0.9743  3.9070  4.5833 70.8719 

# vali_y2<-(target2[-ss])

# distribution of predict proba
hist(p)

# minus value to zero
pr <- ifelse(p<0,0,p)
# pred2<-(p)

# MSE : 27.9085
mse1<-mean((pr-vali_y2)^2) ## 38.21
# pred2<-pr

# *p3 : 0~1사이의 값에 64를 곱한 survival time
# pe3 : p3 중 63 이상은 64로, 3 미만은 3으로 수정
pe3 <- ifelse(p3 > 63,64,p3)
pe3 <- ifelse(p3 < 1,1,p3)
pe3[pe3<3] <- 1

# pe3[pe3<3]<-pe3[pe3<3]^0.3

# pred2 : 총 예측 누적 결제 금액을 예측 잔존일수로 나누어줌
pred2 <- ((pr)/pe3)
pred2[is.na(pred2)]<-0
# pred2 summary
# Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# 0.00000  0.03109  0.10358  0.26332  0.22447 14.05434 

# pred2<-pred2*0.6

pred_label <- train_label
pred_label$survival_time[-ss] <- pe3
pred_label$amount_spent[-ss] <- pred2

# score_function2 <- score_function
score1 <-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss] <- pred2 * 3.5
score2 <- score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*4.5
score3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*5.5
score4<-score_function2(pred_label[-ss],train_label[-ss])

### 단일모델 성능 ###
base_score<-c(acc1,mse1, score1, score2,score3,score4)
base_score %>% round(3)
# base_score : 0.770 30.371 8278.161 11831.689 11736.854 11428.558
# base_score : 0.775 28.115 9227.370 8862.558 8923.950 8812.0149

## 중간 variables 설명 ##
# new_target2 : survival_time과 amount_spent의 곱으로 총 누적 결제금액을 뜻함
# new_target3 : amount_spent가 0이면 new_target3도 0, 그외는 1

# p3 : 0~1사이의 값에 64를 곱한 survival time
# pe3 : p3 중 63 이상은 64로, 3 미만은 3으로 수정
# ppay : binary predict proba by model3, amount_spent가 0이면 0 아니면 1을 예측한 probability 
# pred2 : pred_amount_spent, 총 예측 누적 결제 금액을 예측 잔존일수로 나누어줌

# ppay가 0.05 미만인 acc_id들은 amount spent가 0으로 간주
pred2[ppay< 0.05] <- 0

pred_label$survival_time[-ss] <- pe3
pred_label$amount_spent[-ss] <- pred2
# 예측 amount_spent가 30 이상인 경우 0.75 제곱수를 적용
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30] <- pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
score1 <- score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*3.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75

score2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*4.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75

score3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*5.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
score4<-score_function2(pred_label[-ss],train_label[-ss])

### 단일모델 보정  성능
base_score2<-c(acc1,mse1, score1, score2,score3,score4)
# base_score2 : 0.770 30.371 8289.806 11963.598 12037.251 11728.997
# base_score2 : 0.775 28.286 6376.142 9296.691 9513.048 8623.087

###############################################
############ hyper paramter tuning ############
###############################################
prl<-pred_label[ss,]
trl<-train_label[ss,]

#######################################################################################################
########## Accuracy나 MSE가 아닌 score를 최적화 하도록 bayesian optimization을 이용한 tuning ##########
#######################################################################################################

bayesTuneLGB2 <- function(data, k, ...){
  
  if(k<2) stop(">> k is very small \n")
  
  data <- as.data.frame(data)
  data_y <- data[,"y"]
  data_x <- data[,which(colnames(data)!="y")]
  
  # ...
  rules <- lgb.prepare_rules(data = data_x)$rules
  target_idx   <- which(colnames(data)=="y")
  cat_features <- names(which(sapply(data[,-target_idx], is.factor)))
  
  set.seed(1)
  # library(caret)
  KFolds <- createFolds(1:nrow(data), k = k, list = TRUE, returnTrain = FALSE)        
  
  oof_preds <- rep(NA, nrow(data))
  oof_score <- list()
  for(i in 1:k){
    
    train_idx = unlist(KFolds[-i])
    valid_idx = unlist(KFolds[i])
    
    # dtrain
    dtrain <- lgb.Dataset(
      data = as.matrix(lgb.prepare_rules(data = data_x[train_idx,],  rules = rules)[[1]]), 
      label = data_y[train_idx], 
      colnames = colnames(data_x),
      categorical_feature = cat_features
    )
    
    # dvalid
    dvalid <- lgb.Dataset(
      data = as.matrix(lgb.prepare_rules(data = data_x[valid_idx,],  rules = rules)[[1]]), 
      label = data_y[valid_idx], 
      colnames = colnames(data_x),
      categorical_feature = cat_features
    )
    
    set.seed(1)
    ml_lgb <- lgb.train(
      params = ...,
      data = dtrain,
      valids = list(test = dtrain),
      
      objective = "regression_l2",
      eval = "l2", 
      nrounds = iterations,
      verbose = -1,
      record = TRUE,
      eval_freq = 10,
      learning_rate = learning_rate,
      num_threads = num_threads,
      early_stopping_rounds = early_stopping_rounds
    )
    
    mvalid <- as.matrix(lgb.prepare_rules(data=data_x[valid_idx,], rules=rules)[[1]])
    oof_preds[valid_idx] = predict(ml_lgb, data=mvalid, n=ml_lgb$best_iter)
    
    oof_preds[valid_idx]<-ifelse(oof_preds[valid_idx]<0,0,oof_preds[valid_idx])
    
    prl[valid_idx,]$survival_time<-train_a[valid_idx]
    prl[valid_idx,]$amount_spent<-oof_preds[valid_idx]/train_a[valid_idx]*6.5
    score1<-score_function2(prl[valid_idx,],trl[valid_idx,])
    
    
    oof_score[[i]] = score1
    cat(">> oof_score :", oof_score[[i]], "\n")
  }
  
  list(Score = score_function2(prl,trl), Pred  = oof_preds)
} # end of function

params = list(
  min_data= c(5L,10L, 20L),
  subsample = c(0.5, 1),
  colsample_bytree = c(0.6,0.8, 1),
  num_leaves = c(50L,100L,150L)
)

########
# model2 : 잔존 여부 예측 모델
p = predict(model2, train[ss,])
train_a<-p*64

train_a<-ifelse(train_a>63,64,train_a)
train_a<-ifelse(train_a<1,1,train_a)
hist(train_a)

# train options
kfolds = 3L
early_stopping_rounds = 100L
iterations = 1000L
num_threads = 8
learning_rate = 0.02
init_points = 5L
n_iter = 10L

data<-cbind(train[ss,],new_target2[ss])
colnames(data)[ncol(data)]<-"y"

# hyper parameter optimization
best_params3 <- rBayesianOptimization::BayesianOptimization(
  FUN = function(...){bayesTuneLGB2(data=data,kfolds, ...)},
  bounds = params,
  init_points = init_points,
  n_iter = n_iter,
  acq = "ucb",
  kappa = 2.576,
  eps = 0.0,
  verbose = TRUE)

bi <- which(best_params3$History$Value == best_params3$Best_Value)

best_params3$History[bi,]
ppt <- best_params3$Histor[bi,]
# ppt
# Round min_data          subsample   colsample_bytree num_leaves             Value
#    6        5 0.9999999999999998 0.6000000000000002         55 21115.87616814402
ht2<-as.matrix(ppt[1,-1])

# *k : 1:14
h_list2[[k]]<-ht2
ht <- ht2
save(ht2,file=paste0(rdata_path,"ht",k,".Rdata"))

# load(paste0(rdata_path,"ht1.Rdata"))

# *ss : 7:3 on survival_time
dtrain <- lgb.Dataset(rbind(train[ss,]),label=c(new_target2[ss]))
dvalid <- lgb.Dataset(rbind(train[ss,]),label=new_target2[ss])
valids <- list(test = dvalid)

params <- list(objective = "regression_l2", metric = "l2", device = "gpu")

# use best tuned parameter
model <- lgb.train(params, dtrain, 2000, valids, min_data = ht[1],subsample=ht[2],
                   colsample_bytree=ht[3], num_leaves=ht[4],eval_freq = 500,
                   learning_rate = 0.02, early_stopping_rounds = 100)
p <- predict(model, train[-ss,])
vali_y2 <- (new_target2[-ss])
# vali_y2<-(target2[-ss])
hist(p)
pr<-ifelse(p<0,0,p)
# pred2<-(p)
mse1<-mean((pr-vali_y2)^2) ##38.21, 28.115
# pred2<-pr

##### 하이퍼 파라미터 튜닝 기본 모델 성능 #####
pe3<-ifelse(p3>63,64,p3)
pe3<-ifelse(p3<1,1,p3)
pe3[pe3<3]<-1

pred2<-((pr)/pe3)
pred2[is.na(pred2)]<-0


pred_label$survival_time[-ss]<-pe3
pred_label$amount_spent[-ss]<-pred2
score1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*3.5
score2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*4.5
score3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*5.5
score4<-score_function2(pred_label[-ss],train_label[-ss])
base_score3<-c(acc1,mse1, score1, score2,score3,score4)
# base_score3 : 0.775 28.115 6333.838 8854.190 8911.291 8793.254

####### 하이퍼 파라미터 튜닝 기본 모델 + 보정 성능 #######
pred2[ppay< 0.05] <-0


pred_label$survival_time[-ss]<-pe3
pred_label$amount_spent[-ss]<-pred2
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75

score1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*3.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75

score2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*4.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75

score3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*5.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
# pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 40]<-40
score4<-score_function2(pred_label[-ss],train_label[-ss])
base_score4<-c(acc1,mse1, score1, score2,score3,score4)
# base_score4 : 0.775 28.115 6347.893 8918.217 9146.798 9227.370

######################
### ensemble model ###
######################

## 01. 잔존 여부 예측 모델 ensemble ##
# fact : binary label on target1(survival_time)
model_list<-list()
dt<-train[ss,]
tl<-fact[ss]
p2<-0
accl<-c()
for(p in 1:5){
  # data sampling
  sq<-sample(1:nrow(dt),nrow(dt),replace = T)
  
  dtrain <- lgb.Dataset(rbind(dt[sq,]),label=tl[sq])
  dvalid <- lgb.Dataset(rbind(dt[,]),label=tl[])
  valids <- list(test = dvalid)
  
  params <-  list(objective = "binary", metric = "AUC", device = "gpu")
  model2 <- lgb.train(params, dtrain, 2000, valids,learning_rate = 0.05, early_stopping_rounds = 100,eval_freq = 500)
  model_list[[p]]<-model2
  
  ind<-ind+1
  
  p <- predict(model2, train[-ss,])
  p2 <- p2+p
  p3 <- p2*(64/length(model_list))
  
  enpred<-ifelse(p3>63,64,p3)
  enpred<-ifelse(enpred<1,1,enpred)
  
  predicted <- ifelse((p2)/length(model_list) > 0.5,1,0)
  vali_y <- fact[-ss]
  
  sum(predicted==0 & vali_y==0)/sum( vali_y==0)
  acck <- sum(predicted==vali_y)/length(vali_y);acck
  accl <- c(accl,acck)
  print(accl)
}

# *k : 1:14
acc_list[[k]]<-accl

pp<-list()
p2<-0
for(q in 1:length(model_list)){
  model2<-model_list[[q]]
  p <- predict(model2, train[-ss,])
  p2 <- p2+p
  pp[[q]]<-p
}
pp2<-do.call("cbind",pp)
# dim(pp2) : 12000,5

## pad : 표준편차가 0.03 미만인 것들의 평균 예측값(확률), 4215개
pad <- apply(pp2[apply(pp2,1,sd) < 0.03,],1,mean)
add_index <- apply(pp2,1,sd) < 0.03
# assign binary label, threshold = 0.5
add_label <- round(pad,0)
add_data <- train[-ss,][add_index,] ##### 이후에 학습데이터로 추가할 검증 데이터 
vali_y<-fact[-ss]

# Acc : 0.942111506524318
sum(add_label==vali_y[add_index])/length(add_label)

# p2 : accumulated predict proba
p3 <- p2*(64/length(model_list))
# enpred : final predicted survival time
enpred <- ifelse(p3>63,64,p3)
enpred <- ifelse(enpred<1,1,enpred)

# predicted : p2 to binary
predicted <- ifelse((p2)/length(model_list) > 0.5,1,0)
vali_y <- fact[-ss]

# Acc2 : 0.778
acc2<- sum(predicted==vali_y)/length(vali_y);acc2

## 02. 총 누적결제 금액을 예측하기 위한 모델 ensemble ##
model_list2<-list()
new_target2<-target1*target2 
nt<-new_target2[ss]

p2<-0
dt<-train[ss,]
mse3<-c()
# modeling
for(p in 1:5){
  sq<-sample(1:nrow(dt),nrow(dt),replace = T)
  vv<-sample(1:ncol(dt),ncol(dt)*1)
  dtrain <- lgb.Dataset(rbind(dt[sq,]),label=nt[sq])
  dvalid <- lgb.Dataset(rbind(dt[,]),label=nt[])
  valids <- list(test = dvalid)
  
  params <- list(objective = "regression_l2", metric = "l2", device = "gpu")
  model <- lgb.train(params, dtrain, 2000, valids,learning_rate = 0.02, early_stopping_rounds = 100,eval_freq = 500)
  
  model_list2[[p]]<-model
  
  ind2<-ind2+1
  p = predict(model, train[-ss,])
  p2<-p2+p
  ep<-p2/(length(model_list2))
  
  ep<-ifelse(ep<0,0,ep)
  
  pred2<-(ep)
  vali_y2<-(new_target2[-ss])
  mse2<-mean((pred2-vali_y2)^2) ##38.21
  mse3<-c(mse3,mse2)
  
  print(mse3)
} # end of for

mse_list[[k]]<-mse3

# make predict value
p_list<-list()
p2 <- 0
for(q in 1:length(model_list2)){
  model2<-model_list2[[q]]
  p = predict(model2, train[-ss,])
  p2<-p2+p
  p_list[[q]]<-p
}

ep <- p2/length(model_list2)
ep <- ifelse(ep<0,0,ep)
# ep summary
# Min.          1st Qu.           Median             Mean          3rd Qu.             Max. 
# 0.0000000000000  0.5585259905728  2.5144144246807  3.8974518096744  5.7022009652280 24.9128837591288 

pred2<-(ep)
vali_y2<-(new_target2[-ss])
mse2<-mean((pred2-vali_y2)^2);mse2 ##38.21, 27.93

### 모델 보정을 위한 모델 ###
model_list3<-list()

nt<-target1*target2 
new_target3<-(ifelse(nt>0,1,0))
nt2<-new_target3[ss]

# modeling
for(p in 1:5){
  sq<-sample(1:nrow(dt),nrow(dt),replace = T)
  vv<-sample(1:ncol(dt),ncol(dt)*1)
  dtrain <- lgb.Dataset(rbind(dt[sq,]),label=nt2[sq])
  dvalid <- lgb.Dataset(rbind(dt[,]),label=nt2[])
  valids <- list(test = dvalid)
  
  params <-  list(objective = "binary", metric = "AUC", device = "gpu")
  model3 <- lgb.train(params, dtrain, 2000, valids,  eval_freq = 500,
                      learning_rate = 0.05, early_stopping_rounds = 100)
  model_list3[[p]]<-model3
}

# make predict value
pp<-0
for(q in 1:length(model_list3)){
  model2<-model_list3[[q]]
  p = predict(model2, train[-ss,])
  pp<-pp+p
}

ppay<-pp/length(model_list3)
predicted_p1 = ifelse((ppay)/1 > 0.05,1,0)
vali_y<-new_target3[-ss]

## acc_pay : 0.80
acc_pay<-sum(predicted_p1==vali_y)/length(vali_y)

##### 앙상블 모델 기본 성능 #####
# *ep : 총 누적결제 금액
# *enpred : final predicted survival time
# enpred2 : final 결제금액
enpred2 <- (ep)/enpred

fp2 <- fp2 + enpred2

pred_label$survival_time[-ss] <- enpred
pred_label$amount_spent[-ss] <- enpred2
escore1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*3.5
escore2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*4.5
escore3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*5.5
escore4<-score_function2(pred_label[-ss],train_label[-ss])

en_score <- c(acc2,mse2,escore1,escore2,escore3,escore4)
# en_score : 0.778 27.932 6967.399 10485.388 10514.110 10279.925

###### 앙상블 모델 기본 +보정 성능 ######
ep[ppay< 0.05] <-0
enpred2<-(ep)/enpred

fp2<-fp2+enpred2

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*3.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*4.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*5.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore4<-score_function2(pred_label[-ss],train_label[-ss])

en_score2<-c(acc2,mse2,escore1,escore2,escore3,escore4)
# en_score2 :  0.778 27.932  6975.224 10531.214 10341.094 10025.062


#############################################
### ensemble 모델 +  하이퍼 파라미터 튜닝 ###
#############################################

## 01. 총 누적결제 금액을 예측하기 위한 모델 ensemble + tuning ##
model_list2<-list()
new_target2<-target1*target2 
nt <- new_target2[ss]

p2 <- 0
dt<-train[ss,]
mse3<-c()

# modeling
for(p in 1:5){
  sq<-sample(1:nrow(dt),nrow(dt),replace = T)
  vv<-sample(1:ncol(dt),ncol(dt)*1)
  dtrain <- lgb.Dataset(rbind(dt[sq,]),label=nt[sq])
  dvalid <- lgb.Dataset(rbind(dt[,]),label=nt[])
  valids <- list(test = dvalid)
  
  params <- list(objective = "regression_l2", metric = "l2", device = "gpu")
  model <- lgb.train(params, dtrain, 2000, valids, min_data = ht[1],subsample=ht[2],
                     colsample_bytree=ht[3], num_leaves=ht[4],eval_freq = 500,
                     learning_rate = 0.02, early_stopping_rounds = 100)
  model_list2[[p]]<-model
  
  ind2<-ind2+1
  p <- predict(model, train[-ss,])
  p2 <- p2+p
  ep<-p2/(length(model_list2))
  ep<-ifelse(ep<0,0,ep)
  
  pred2 <- (ep)
  vali_y2<-(new_target2[-ss])
  mse2<-mean((pred2-vali_y2)^2) ##38.21
  mse3<-c(mse3,mse2)
  
  print(mse3)
}
mse_list[[k]]<-mse3

# predict
p_list<-list()
p2<-0
for(q in 1:length(model_list2)){
  model2<-model_list2[[q]]
  p = predict(model2, train[-ss,])
  p2<-p2+p
  p_list[[q]]<-p
}
pv1<-do.call("cbind",p_list)
back_pv1 <- pv1
# back_pv1 %>% dim : 12000 5

# 0.05 미만 표준편차 : 568개
sum(apply(pv1,1,sd) < 0.05)

# 표준편차가 0.05 미만인 것 재학습 사용
new2_index <- sqrt(apply(pv1,1,var)) < 0.05
# 평균값을 label로 할당
new2 <- apply(pv1[apply(pv1,1,sd) < 0.05,],1,mean)
new2 <- ifelse(new2 < 0,0,new2)
add_data2 <- train[-ss,][new2_index,] # ## 이후에 학습에 추가할 검증데이터

ep <- p2/length(model_list2)
ep <- ifelse(ep<0,0,ep)
# ep summary
# Min.          1st Qu.           Median             Mean          3rd Qu.             Max. 
# 0.0000000000000  0.5713263600683  2.5200534469815  3.9518486236877  5.8129362362029 22.6366749935533 
pred2 <- (ep)
vali_y2<-(new_target2[-ss])
mse2<-mean((pred2-vali_y2)^2) ##38.21 27.907

############ 앙상블 모델 하이퍼파라미터 튜닝 기본 성능 #############
# *ep : 총 누적결제 금액, 갱신
# *enpred : final predicted survival time, 미갱신
enpred2<-(ep)/enpred

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2
escore1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*3.5
escore2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*4.5
escore3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*6
escore4<-score_function2(pred_label[-ss],train_label[-ss])

en_score3<-c(acc2,mse2,escore1,escore2,escore3,escore4)
# en_score3 :  0.778 27.908  6842.658 10354.700 10462.462 10073.679

############# 앙상블 모델 하이퍼파라미터 튜닝 기본 + 보정 성능 #############
ep[ppay< 0.05] <-0
enpred2<-(ep)/enpred

fp2<-fp2+enpred2

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*3.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*4.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*5.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore4<-score_function2(pred_label[-ss],train_label[-ss])

en_score4<-c(acc2,mse2,escore1,escore2,escore3,escore4)
# en_score4 : 0.778 27.908  6855.721 10425.616 10284.643 10212.256

#################################################################################################################################
##################### 신뢰성이 높은 검증데이터를 다시 학습데이터로 재사용 하는 기법  + 하이퍼 파라미터 튜닝 #####################
#################################################################################################################################

# add data : 잔존 여부 예측 모델 ensemble, 0.03미만, 4215개
dtrain <- lgb.Dataset(rbind(train[ss,],add_data),label=c(fact[ss],add_label))
dvalid <- lgb.Dataset(rbind(train[ss,]),label=fact[ss])
valids <- list(test = dvalid)

params <-  list(objective = "binary", metric = "AUC", device = "gpu")
model2 <- lgb.train(params, dtrain, 2000, valids, eval_freq = 500,
                    learning_rate = 0.05, early_stopping_rounds = 100)

p <- predict(model2, train[-ss,])
p3 <- p*64

pe3<-ifelse(p3>63,64,p3)
pe3<-ifelse(p3<1,1,p3)

predicted = ifelse((p)/1 > 0.5,1,0)
vali_y<-fact[-ss]
# Check True Negative : 0.7169
sum(predicted==0 & vali_y==0)/sum(vali_y==0)
# Check True Positive : 0.8299
sum(predicted==1 & vali_y==1)/sum(vali_y==1)

# Acc : 0.7797
acc3<-sum(predicted==vali_y)/length(vali_y);acc3

# add_data2 : 총 누적 결제금액 예측모델 ensemble + parameter tuning, 0.05미만, 568개
dtrain <- lgb.Dataset(rbind(train[ss,],add_data2),label=c(new_target2[ss],new2))
dvalid <- lgb.Dataset(rbind(train[ss,]),label=new_target2[ss])
valids <- list(test = dvalid)

# use best tuned parameter
params <- list(objective = "regression_l2", metric = "l2", device = "gpu")
model <- lgb.train(params, dtrain, 2000, valids, min_data = ht[1],subsample=ht[2],
                   colsample_bytree=ht[3], num_leaves=ht[4],eval_freq = 500,
                   learning_rate = 0.02, early_stopping_rounds = 100)
p <- predict(model, train[-ss,])
vali_y2 <- (new_target2[-ss])
pr <- ifelse(p<0,0,p)

mse3<-mean((pr-vali_y2)^2); mse3 ##28.06
pred2 <- pr

# 예측된 총 누적 결제 금액을 예측된 잔존 일수로 나누어 줌
pe3<-ifelse(p3>63,64,p3)
pe3<-ifelse(p3<1,1,p3)
pe3[pe3<3]<-1

pred2<-((pr)/pe3)
pred2[is.na(pred2)]<-0
# pred2 summary
# Min.           1st Qu.            Median              Mean           3rd Qu.              Max. 
# 0.00000000000000  0.03053827776846  0.10227286978131  0.29371400745524  0.22587211797465 12.70255549076843 

######## 재학습 + 하이퍼파라미터 튜닝 단일모델 기본 성능
pred_label$survival_time[-ss]<-pe3
pred_label$amount_spent[-ss]<-pred2
score1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*3.5
score2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*4.5
score3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*5.5
score4<-score_function2(pred_label[-ss],train_label[-ss])
base_score5<-c(acc3,mse3, score1, score2,score3,score4)
# base_score5 : 0.780  28.063 7216.137 10185.875 10044.965  9573.690

########재학습 + 하이퍼파라미터 튜닝 단일모델 기본 + 보정 성능
pred2[ppay< 0.05] <-0

pred_label$survival_time[-ss]<-pe3
pred_label$amount_spent[-ss]<-pred2
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75

score1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*3.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75

score2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*4.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75

score3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*5.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
score4<-score_function2(pred_label[-ss],train_label[-ss])
base_score6<-c(acc1,mse1, score1, score2,score3,score4)
# base_score6 : 0.775 28.1157202.651 10201.081 10105.484 9671.407


####################################################################################################################
########## 신뢰성이 높은 검증데이터를 다시 학습데이터로 재사용 하는 기법  + 앙상블 + 하이퍼 파라미터 튜닝 ##########
####################################################################################################################

## 01. 잔존 여부 예측 모델 ensemble ##
model_list<-list()
dt<-rbind(train[ss,],add_data)
tl<-c(fact[ss],add_label)
p2<-0
accl<-c()

# modeling
for(p in 1:5){
  sq<-sample(1:nrow(dt),nrow(dt),replace = T)
  dtrain <- lgb.Dataset(rbind(dt[sq,]),label=tl[sq])
  dvalid <- lgb.Dataset(rbind(dt[,]),label=tl[])
  valids <- list(test = dvalid)
  
  params <-  list(objective = "binary", metric = "AUC", device = "gpu")
  model2 <- lgb.train(params, dtrain, 2000, valids, eval_freq = 500,
                      learning_rate = 0.05, early_stopping_rounds = 100)
  
  model_list[[p]]<-model2
  ind<-ind+1
  
  p <- predict(model2, train[-ss,])
  p2<-p2+p
  p3<-p2*(64/length(model_list))
  
  enpred<-ifelse(p3>63,64,p3)
  enpred<-ifelse(enpred<1,1,enpred)
  
  predicted = ifelse((p2)/length(model_list) > 0.5,1,0)
  vali_y<-fact[-ss]
  
  sum(predicted==0 & vali_y==0)/sum( vali_y==0)
  acck<- sum(predicted==vali_y)/length(vali_y);acck
  accl<-c(accl,acck)
  print(accl)
} # end of for
acc_list[[k]]<-accl

# predict
pp<-list()
p2<-0
for(q in 1:length(model_list)){
  model2<-model_list[[q]]
  p = predict(model2, train[-ss,])
  p2<-p2+p
  pp[[q]]<-p
}

pp2<-do.call("cbind",pp)
p3<-p2*(64/length(model_list))
enpred<-ifelse(p3>63,64,p3)
enpred<-ifelse(enpred<1,1,enpred)
fp<-fp+enpred

# 0:10 = 1:64

predicted = ifelse((p2)/length(model_list) > 0.5,1,0)
vali_y<-fact[-ss]

# 0.714
sum(predicted==0 & vali_y==0)/sum(vali_y==0)
# 0.8279
sum(predicted==1 & vali_y==1)/sum(vali_y==1)
# Acc : 0.7774
acc2<- sum(predicted==vali_y)/length(vali_y);acc2


## 02. 총 누적결제 금액을 예측하기 위한 모델 ensemble + tuning ##
model_list2<-list()
new_target2<-target1*target2 
nt<-new_target2[ss]

p2<-0
dt<-rbind(train[ss,],add_data2)
nt<-c(nt,new2)
mse3<-c()
for(p in 1:5){
  sq<-sample(1:nrow(dt),nrow(dt),replace = T)
  vv<-sample(1:ncol(dt),ncol(dt)*1)
  dtrain <- lgb.Dataset(rbind(dt[sq,]),label=nt[sq])
  dvalid <- lgb.Dataset(rbind(dt[,]),label=nt[])
  valids <- list(test = dvalid)
  params <- list(objective = "regression_l2", metric = "l2", device = "gpu")
  
  # use best tuned hyper parameter
  model <- lgb.train(params, dtrain, 2000, valids, min_data = ht[1],subsample=ht[2],
                     colsample_bytree=ht[3], num_leaves=ht[4],eval_freq = 500,
                     learning_rate = 0.02, early_stopping_rounds = 100)
  model_list2[[p]]<-model
  
  ind2<-ind2+1
  p = predict(model, train[-ss,])
  p2<-p2+p
  ep<-p2/(length(model_list2))
  
  ep<-ifelse(ep<0,0,ep)
  
  pred2<-(ep)
  vali_y2<-(new_target2[-ss])
  mse2<-mean((pred2-vali_y2)^2) ##38.21
  mse3<-c(mse3,mse2)
}
mse_list[[k]]<-mse3

p_list<-list()
p2<-0
for(q in 1:length(model_list2)){
  model2<-model_list2[[q]]
  p = predict(model2, train[-ss,])
  p2<-p2+p
  p_list[[q]]<-p
}


ep<-p2/length(model_list2)
ep<-ifelse(ep<0,0,ep)

pred2<-(ep)
vali_y2<-(new_target2[-ss])
mse2<-mean((pred2-vali_y2)^2) ##38.21

enpred[enpred<3]<-1
enpred2<-(ep)/enpred


# 신뢰성이 높은 검증데이터를 다시 학습데이터로 재사용 하는 기법  + 하이퍼 파라미터 튜닝  기본 성능
pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2
escore1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*3.5
escore2<-score_function2(pred_label[-ss],train_label[-ss])
escore2

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*4.5
escore3<-score_function2(pred_label[-ss],train_label[-ss])
escore3

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*6
escore4<-score_function2(pred_label[-ss],train_label[-ss])
escore4


en_score5<-c(acc2,mse2,escore1,escore2,escore3,escore4)


# 신뢰성이 높은 검증데이터를 다시 학습데이터로 재사용 하는 기법  + 하이퍼 파라미터 튜닝 기본 + 보정 성능
ep[ppay< 0.05] <-0
enpred2<-(ep)/enpred

fp2<-fp2+enpred2

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*3.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore2<-score_function2(pred_label[-ss],train_label[-ss])
escore2

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*4.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore3<-score_function2(pred_label[-ss],train_label[-ss])
escore3

pred_label$survival_time[-ss]<-enpred
pred_label$amount_spent[-ss]<-enpred2*5.5
pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]<-pred_label$amount_spent[-ss][pred_label$amount_spent[-ss] > 30]^0.75
escore4<-score_function2(pred_label[-ss],train_label[-ss])
escore4

c(acc2,mse2,escore1,escore2,escore3,escore4)
en_score6<-c(acc2,mse2,escore1,escore2,escore3,escore4)



final_score<-rbind(final_score,c(base_score,base_score2,base_score3,base_score4,base_score5,base_score6))
final_score2<-rbind(final_score2,c(en_score,en_score2,en_score3,en_score4,en_score5,en_score6))

#####단일모델
cat("\n기본",base_score)
cat("\n기본+보정",base_score2)
cat("\n기본+튜닝",base_score3)
cat("\n기본+튜닝+보정",base_score4)
cat("\n재학습+튜닝",base_score5)
cat("\n재학습+튜닝+보정",base_score6)

####ensemble
cat("\nE기본",en_score)
cat("\nE기본+보정",en_score2)
cat("\nE기본+튜닝",en_score3)
cat("\nE기본+튜닝+보정",en_score4)
cat("\nE재학습+튜닝",en_score5)
cat("\nE재학습+튜닝+보정",en_score6)


################
## 주어진 학습데이터 내에서
## 앙상블 > 단일모델
## 보정 > 기본
## 하이퍼파라미터 튜닝> 기본값
## 재학습 > 기본 
## 최종적으로 앙상블 + 튜닝 + 보정 -> 다시 재학습 하는 것이 가장 좋음 

write.csv(final_score,paste0(csv_path, "validationv_Base_재학습_",k,".csv"),row.names=F)
write.csv(final_score2,paste0(csv_path,"validationv_ensemble_재학습_",k,".csv"),row.names=F)

} # end of for loop
rm(list = ls())
} # end of whole_for loop

##################### feature selection

train<-cbind(base_data,serv_df,pled_data1,pvp_data1,diff_time_df,diff_pay_df,train_data1,train_data2,train_data3,train_data4,strain_data1,strain_data2) # 0.742
train<-as.matrix(train[match(train_label$acc_id,taid),])
train[(is.na(train))]<-0
varn<-rep(0,ncol(train))
varn2<-rep(0,ncol(train))
veve<-rep(0,ncol(train))
veve2<-rep(0,ncol(train))
for( t in 1:1000){
  
  
  fs<-fact[ss]
  
  dt<-train[ss,]
  
  vv<-sample(1:ncol(train),ncol(train)*0.25)
  dtrain <- lgb.Dataset(rbind(train[ss,vv]),label=c(fact[ss]))
  dvalid <- lgb.Dataset(rbind(train[ss,vv]),label=fact[ss])
  valids <- list(test = dvalid)
  head(train)
  
  
  params <-  list(objective = "binary", metric = "AUC", device = "gpu")
  
  model2 <- lgb.train(params, dtrain, 10, valids,  eval_freq = 500,
                      learning_rate = 0.05, early_stopping_rounds = 100)
  
  
  
  
  p = predict(model2, train[-ss,vv])
  p3<-p*64
  0.5*64
  pe3<-ifelse(p3>63,64,p3)
  pe3<-ifelse(p3<1,1,p3)
  
  predicted = ifelse((p)/1 > 0.5,1,0)
  vali_y<-fact[-ss]
  sum(predicted==0 & vali_y==0)/sum( vali_y==0)
  
  sum(predicted==1 & vali_y==1)/sum( vali_y==1)
  
  acc1<-sum(predicted==vali_y)/length(vali_y) ## acc1
  acc1
  cat("\n",t,"-",acc1)
  veve[vv]<-acc1
  varn<-rbind(varn,veve)
  
  
  new_target2<-target1*target2 
  dtrain <- lgb.Dataset(rbind(train[ss,vv]),label=c(new_target2[ss]))
  dvalid <- lgb.Dataset(rbind(train[ss,vv]),label=new_target2[ss])
  
  valids <- list(test = dvalid)
  
  
  params <- list(objective = "regression_l2", metric = "l2" , device = "gpu")
  model2 <- lgb.train(params, dtrain, 10, valids,  eval_freq = 500,
                      learning_rate = 0.05, early_stopping_rounds = 100)
  
  p = predict(model2, train[-ss,vv])
  vali_y2<-(new_target2[-ss])
  
  pr<-ifelse(p<0,0,p)
  # pred2<-(p)
  mse<-mean((pr-vali_y2)^2) ##38.21
  
  veve2[vv]<-mse
  varn2<-rbind(varn2,veve2)
}


save(varn2,file="varn2.RData")
load(paste0(rdata_path,"varn2.RData"))

save(varn,file="varn.RData")
load(paste0(rdata_path,"varn.RData"))

hist(apply(varn,2,mean,na.rm=T))

varn %>% dim
varn2 %>% dim
varn[1:6,1:6]
varn2[1:6,1:6]

varn[varn==0]<-NA

hist(apply(varn,2,mean,na.rm=T))
summary(apply(varn,2,mean,na.rm=T))
colnames(train)[which(apply(varn,2,mean,na.rm=T)<0.7668)]
colnames(train)[which(apply(varn,2,mean,na.rm=T)>0.7668)]
bbas<-which(apply(varn,2,mean,na.rm=T)<0.7665)

no_var<-colnames(train)[bbas]
save(no_var,file="no_var_.RData")

varn2[varn2==0]<-NA
summary(apply(varn2,2,mean,na.rm=T))
colnames(train)[which(apply(varn2,2,mean,na.rm=T)<29.57)]
colnames(train)[which(apply(varn2,2,mean,na.rm=T)>29.57)]
bbas2<-which(apply(varn2,2,mean,na.rm=T)>29.75)
bbas2
no_var2<-colnames(train)[bbas2]
no_var %>% length
no_var2 %>% length
save(no_var2,file="no_var2_.RData")



######################모든 feature다 넣은 기본 모델의 성능 

dtrain <- lgb.Dataset(rbind(train[ss,]),label=c(fact[ss]))
dvalid <- lgb.Dataset(rbind(train[ss,]),label=fact[ss])
valids <- list(test = dvalid)
params <-  list(objective = "binary", metric = "AUC" )

model2 <- lgb.train(params, dtrain, 2000, valids,  eval_freq = 500,
                    learning_rate = 0.05, early_stopping_rounds = 100)



p = predict(model2, train[-ss,])
p3<-p*64
0.5*64
pe3<-ifelse(p3>63,64,p3)
pe3<-ifelse(p3<1,1,p3)

predicted = ifelse((p)/1 > 0.5,1,0)
vali_y<-fact[-ss]
sum(predicted==0 & vali_y==0)/sum( vali_y==0)

sum(predicted==1 & vali_y==1)/sum( vali_y==1)

acc1<-sum(predicted==vali_y)/length(vali_y) ## acc1
acc1





new_target2<-target1*target2 
dtrain <- lgb.Dataset(rbind(train[ss,]),label=c(new_target2[ss]))
dvalid <- lgb.Dataset(rbind(train[ss,]),label=new_target2[ss])

valids <- list(test = dvalid)


params <- list(objective = "regression_l2", metric = "l2" )
model2 <- lgb.train(params, dtrain, 500, valids,  eval_freq = 500,
                    learning_rate = 0.05, early_stopping_rounds = 100)

p = predict(model2, train[-ss,])
vali_y2<-(new_target2[-ss])

pr<-ifelse(p<0,0,p)
# pred2<-(p)
mse<-mean((pr-vali_y2)^2) ##38.21
mse

pe3<-ifelse(p3>63,64,p3)
pe3<-ifelse(p3<1,1,p3)
pe3[pe3<3]<-1

# pe3[pe3<3]<-pe3[pe3<3]^0.3
pred2<-((pr)/pe3)
# pred2<-pred2*0.6
pred2[is.na(pred2)]<-0
pred_label$survival_time[-ss]<-pe3
pred_label$amount_spent[-ss]<-pred2
score1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*3.5
score2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*4.5
score3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*5.5
score4<-score_function2(pred_label[-ss],train_label[-ss])

no_feat<-c(acc1,mse, score1, score2,score3,score4)
no_feat




######################선택된 feature만 넣은 기본 모델의 성능 

dtrain <- lgb.Dataset(rbind(train[ss,-bbas]),label=c(fact[ss]))
dvalid <- lgb.Dataset(rbind(train[ss,-bbas]),label=fact[ss])
valids <- list(test = dvalid)
params <-  list(objective = "binary", metric = "AUC" )

model2 <- lgb.train(params, dtrain, 2000, valids,  eval_freq = 500,
                    learning_rate = 0.05, early_stopping_rounds = 100)



p = predict(model2, train[-ss,-bbas])
p3<-p*64
0.5*64
pe3<-ifelse(p3>63,64,p3)
pe3<-ifelse(p3<1,1,p3)

predicted = ifelse((p)/1 > 0.5,1,0)
vali_y<-fact[-ss]
sum(predicted==0 & vali_y==0)/sum( vali_y==0)

sum(predicted==1 & vali_y==1)/sum( vali_y==1)

racc1<-sum(predicted==vali_y)/length(vali_y) ## acc1
racc1





new_target2<-target1*target2 
dtrain <- lgb.Dataset(rbind(train[ss,-bbas2]),label=c(new_target2[ss]))
dvalid <- lgb.Dataset(rbind(train[ss,-bbas2]),label=new_target2[ss])

valids <- list(test = dvalid)


params <- list(objective = "regression_l2", metric = "l2" )
model2 <- lgb.train(params, dtrain, 500, valids,  eval_freq = 500,
                    learning_rate = 0.05, early_stopping_rounds = 100)

p = predict(model2, train[-ss,-bbas2])
vali_y2<-(new_target2[-ss])

pr<-ifelse(p<0,0,p)
# pred2<-(p)
rmse<-mean((pr-vali_y2)^2) ##38.21
rmse
mse<-rmse




pe3<-ifelse(p3>63,64,p3)
pe3<-ifelse(p3<1,1,p3)
pe3[pe3<3]<-1

# pe3[pe3<3]<-pe3[pe3<3]^0.3
pred2<-((pr)/pe3)
# pred2<-pred2*0.6
pred2[is.na(pred2)]<-0
pred_label$survival_time[-ss]<-pe3
pred_label$amount_spent[-ss]<-pred2
score1<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*3.5
score2<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*4.5
score3<-score_function2(pred_label[-ss],train_label[-ss])

pred_label$amount_spent[-ss]<-pred2*5.5
score4<-score_function2(pred_label[-ss],train_label[-ss])

feat_score<-c(racc1,rmse, score1, score2,score3,score4)
feat_score




