from src.wrapper import Wrapper

tweets = [
  "Harus diakui, joke Chris Rock jahat & gak lucu-lucu amat. Tapi Will Smith sampe naik panggung & nampar orang lagi perform itu lebih kelewatan. Menurut gw sih gitu, tapi yaudahlah ya. 🤪",
  "terlepas dari banyak kekurangan di sanasini gw sih sebagai warga indonesia tetep bangga negara kita udah bisa menyelenggarakan moto gp 💙",
  "terlepas dari banyak kekurangan di sanasini gw sih sebagai warga indonesia tetep bangga negara kita udah bisa menyelenggarakan moto gp 💙",
  "terlepas dari banyak kekurangan di sanasini gw sih sebagai warga indonesia tetep bangga negara kita udah bisa menyelenggarakan moto gp 💙"
]

retweets = [
  "si Chris Rock mang ga ada lawakan lain gitu ya selain prihal penyakit dan penampilan gundul istrinya will smith?",
  "bangga donh 😊",
  "di 1996 juga udah di selenggarakan moto gp koh",
  "bangga tapi tidak dengan pawang hujannya🤣"
]

if __name__ == '__main__':
  model_wrapper = Wrapper()
  pred = model_wrapper.predict(tweets, retweets)
  print(pred)