def get_param_list(start, increment, limit):
  data = [start]
  for i in range(limit):
    data.append(data[-1] + increment)

  return data


parameters = {"loss": ["deviance", "exponential"], 
              "learning_rate": get_param_list(0.1, 0.1, 9), 
              "n_estimators": get_param_list(70, 10, 50)}


print(parameters["learning_rate"][0], parameters["learning_rate"][-1])
print(parameters["n_estimators"][0], parameters["n_estimators"][-1])