def recommend_action(churn_prob):
    if churn_prob >= 0.75:
        return "Immediate retention call & special offer"
    elif churn_prob >= 0.40:
        return "Offer discount or loyalty benefits"
    else:
        return "Upsell or upgrade plan"
