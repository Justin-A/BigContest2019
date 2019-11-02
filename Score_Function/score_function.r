

# ------------------------------------------------------------------------ #
# INPUT
#    predicted_label : 예측 답안지 파일 경로 
#    actual_label    : 실제 답안지 파일 경로
#
# OUTPUT             : 유저 기대이익 총합 
#
# 필요 라이브러리     : 없음
#
# - 예측 답안지를 실제 답안과 비교하여 유저 기대이익 총합을 계산하는 함수
# - 함수의 계산방식은 문제 설명서에 기술된 기대이익 산출식과 동일
# ------------------------------------------------------------------------ #

score_function = function( predicted_label, actual_label )
{
  options(digits=16) # 총 16자리까지 사용하여 연산하도록 설정
  
  predicted_label = read.csv(predicted_label, header = TRUE) # 예측 답안 파일 불러오기
  actual_label = read.csv(actual_label, header = TRUE)       # 실제 답안 파일 불러오기 
  
  ordered_predicted_label = predicted_label[order(predicted_label$acc_id),] # 예측 답안을 acc_id 기준으로 정렬 
  ordered_actual_label = actual_label[order(actual_label$acc_id), ]         # 실제 답안을 acc_id 기준으로 정렬
  
  if( all(ordered_predicted_label$acc_id != ordered_actual_label$acc_id) )
  {
    stop("acc_id of predicted and actual label does not match", call. = FALSE)  # 예측 답안의 acc_id와 실제 답안의 acc_id가 다른 경우 에러처리 
    
  } else {
    predicted_label = ordered_predicted_label
    actual_label = ordered_actual_label
    
    predicted_survival_time = predicted_label$survival_time
    predicted_amount_spent = predicted_label$amount_spent
    
    actual_survival_time = actual_label$survival_time
    actual_amount_spent = actual_label$amount_spent
    
    additional_survival_time = ifelse( predicted_survival_time == 64 | actual_survival_time == 64,                                  # 추가 생존기간 계산
                                       0,
                                       30 * exp(1) ^ (-( (actual_survival_time - predicted_survival_time) ^ 2) / (2 * (15 ^ 2)))
      )
    
    cost = ifelse( predicted_amount_spent == 0 | predicted_survival_time == 64,                                                     # 비용 계산
                           0,
                           0.01 * 30 * predicted_amount_spent
                           )
    optimal_cost = 0.01 * 30 * actual_amount_spent                                                                                  # 적정 비용 계산
    
    response_rate = ifelse( optimal_cost == 0 | (cost / optimal_cost) < 0.1, 0 ,                                                    # 반응률 계산
                            ifelse ( cost / optimal_cost >= 1, 1,
                              cost / (0.9 * optimal_cost) - 1/9
                            )
                          )
  
    additional_sales = additional_survival_time * actual_amount_spent                                                               # 잔존 가치 계산
     
    user_value = sum( additional_sales * response_rate - cost)                                                                      # 유저별 기대 이익 계산 및 합산
    
    return(user_value)                 
  }
}



