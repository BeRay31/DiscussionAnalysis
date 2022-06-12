from src.wrapper import Wrapper

tweets = [
  "Terlepas dari banyak kekurangan di sana-sini, gw sih sebagai warga Indonesia tetep bangga negara kita udah bisa menyelenggarakan Moto GP. 💙",
  "Harus diakui, joke Chris Rock jahat & gak lucu-lucu amat. Tapi Will Smith sampe naik panggung & nampar orang lagi perform itu lebih kelewatan. Menurut gw sih gitu, tapi yaudahlah ya. 🤪",
  "DPR Setujui Harga Pertamax Naik Rp 16.000 per Liter"
]

retweets = [
  "Apalg org Lombok Bang, bahagiaaaa akhirnya bs hadir di mata dunia di ajang bergengsi, rejeki juga buat kami yg lama libur wisatawan selama pandemi",
  "si Chris Rock mang ga ada lawakan lain gitu ya selain prihal penyakit dan penampilan gundul istrinya will smith?",
  "Mereka Serentak sepakat MEMERAS Rakyat."
]

if __name__ == '__main__':
  model_wrapper = Wrapper()
  pred = model_wrapper.predict(tweets, retweets)
  print(pred)