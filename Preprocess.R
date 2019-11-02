# justin_a@yonsei.ac.kr
#rm(list=ls())
#setwd("D:\\개인폴더\\대회\\빅콘테스트\\2019\\2019_빅콘테스트_챔피언스리그_데이터")
### setwd
for (tmp in (1:14)){
  setwd("~/bigcon/data")}

## load score function
source("~/bigcon/loss function/score_function.r")
score_function2 <- score_function

### load pacakges
if(!require(dplyr)) install.packages("dplyr"); library(dplyr)
if(!require(caret)) install.packages("caret"); library(caret)
if(!require(data.table)) install.packages("data.table"); library(data.table)
if(!require(reshape)) install.packages("reshape"); library(reshape)
if(!require(stringr)) install.packages("stringr"); library(stringr)
if(!require(LaplacesDemon)) install.packages("LaplacesDemon"); library(LaplacesDemon)
if(!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if(!require(ggplot2)) install.packages("ggplot2"); library(ggplot2)
if(!require(viridis)) install.packages("viridis"); library(viridis)
if(!require(stringr)) install.packages("stringr"); library(stringr)
if(!require(lightgbm)) install.packages("lightgbm"); library(lightgbm)
#if(!require(Metrics)) install.packages("Metrics"); library(Metrics)
if(!require(rBayesianOptimization)) install.packages("rBayesianOptimization"); library(rBayesianOptimization)

### set path
main_path <- "~/bigcon/"
model_path <- paste0(main_path,"models/")
raw_path  <- paste0(main_path,"data/")
csv_path  <- paste0(main_path,"csv/")
profiles_path <- paste0(main_path,"profile_data/")
rdata_path <- paste0(main_path,"lgt/train2/")

### function
read_fread <- function(filename) {
  output <- fread(paste0(raw_path,filename))
  return(output)
}

### load data
## whole raw data in working directory
list.files(raw_path)

train_label <-read_fread('train_label.csv')
train_ac <-read_fread("train_activity.csv")
train_pa <-read_fread("train_payment.csv")
train_com <-read_fread("train_combat.csv")
train_tr <-read_fread("train_trade.csv")
train_pl <-read_fread("train_pledge.csv")
taid<-unique(train_ac$acc_id)

## load rdata
load(paste0(rdata_path, "vvtrain_data1.RData"))
load(paste0(rdata_path, "vvtrain_data2.RData"))
load(paste0(rdata_path, "vvtrain_data3.RData"))
load(paste0(rdata_path, "vvtrain_data4.RData"))
load(paste0(rdata_path, "vvstrain_data1.RData"))
load(paste0(rdata_path, "vvstrain_data2.RData"))
load(paste0(rdata_path, "vvbase_data.RData"))
load(paste0(rdata_path, "vvdiff_time_df.RData"))
load(paste0(rdata_path, "vvdiff_pay_df.RData"))
load(paste0(rdata_path, "vvpled_data1.RData"))
load(paste0(rdata_path, "vvpvp_data1.RData"))
load(paste0(rdata_path, "vvserv_df.RData"))

## set target
target1<-train_label$survival_time
target2<-train_label$amount_spent

## index split, 7:3 on survival_time
ss <- sample(1:length(target1),length(target1)*0.7)

acc_list<-list()
mse_list<-list()
final_score<-NULL
final_score2<-NULL
f_model<-list()
f_model2<-list()
ind<-1
ind2<-1
fp<-0
fp2<-0
k<-1
h_list<-list()
h_list2<-list()
enlist<-list()
for(k in 1:14){
  # train1<-cbind(train_data1) #0.742
  head(base_data)
  base_data<-data.frame(base_data)
  # last_pay divied by total_pay
  base_data$V1<- base_data$last_pay/base_data$total_pay
  # last_pay divied by cnt_pay
  base_data$V2<- base_data$last_pay/base_data$cnt_pay
  # total_pay divied by num_char (*num_char = number of char_id)
  base_data$V3<- base_data$total_pay/base_data$num_char
  # total_pay divided by total_day (*total_day = day of connections)
  base_data$V4<-base_data$total_pay/base_data$total_day
  base_data[is.na(base_data)]<-0
  
  ## train1~6 : base_feature + 1 core feature
  # *base feature : base_data, serv_df, pled_data1, pvp_data1
  train1<-cbind(base_data,serv_df,pled_data1,pvp_data1,train_data1)
  train2<-cbind(base_data,serv_df,pled_data1,pvp_data1,train_data2)
  train3<-cbind(base_data,serv_df,pled_data1,pvp_data1,train_data3)
  train4<-cbind(base_data,serv_df,pled_data1,pvp_data1,train_data4)
  train5<-cbind(base_data,serv_df,pled_data1,pvp_data1,strain_data1)
  train6<-cbind(base_data,serv_df,pled_data1,pvp_data1,strain_data2)
  
  ## train7~12 : base_feature + diff_feature + 1 core feature
  # *diff_feature : diff_time_df, diff_pay_df
  # final use
  train7<-cbind(base_data,serv_df,pled_data1,pvp_data1,diff_time_df,diff_pay_df,train_data1)
  train8<-cbind(base_data,serv_df,pled_data1,pvp_data1,diff_time_df,diff_pay_df,train_data2)
  train9<-cbind(base_data,serv_df,pled_data1,pvp_data1,diff_time_df,diff_pay_df,train_data3)
  train10<-cbind(base_data,serv_df,pled_data1,pvp_data1,diff_time_df,diff_pay_df,train_data4)
  train11<-cbind(base_data,serv_df,pled_data1,pvp_data1,diff_time_df,diff_pay_df,strain_data1)
  train12<-cbind(base_data,serv_df,pled_data1,pvp_data1,diff_time_df,diff_pay_df,strain_data2)
  
  ## train13 : base_feature + diff_feature + 2 core features
  train13<-cbind(base_data,serv_df,pled_data1,pvp_data1,diff_time_df,diff_pay_df,train_data1,train_data2)
  ## train14 : base_feature + diff_feature + 4 core features
  train14<-cbind(base_data,serv_df,pled_data1,pvp_data1,diff_time_df,diff_pay_df,train_data1,train_data2,strain_data1,strain_data2)
  
  ## use eval function for loop modeling
  indexxx <- paste0("train","<-(train",k,")")
  eval(parse(text=indexxx)) # train <- train1
  
  # preproc na value
  train[(is.na(train))] <- 0
  
  # check inf value in train
  sum((train==Inf))
  
  # data frame to matrix
  train <- as.matrix(train[match(train_label$acc_id,taid),])
  



