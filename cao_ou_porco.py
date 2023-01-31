from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Features(1 sim, 2 não)
# Pelo longo?
# Perna curta?
# Faz auau
porco1 = [0,1,0]
porco2 = [0,1,1]
porco3 = [1,1,0]

cachorro1 = [0,1,1]
cachorro2 = [1,0,1]
cachorro3 = [1,1,1]

# 1 => porco, 2 => cachorro
train_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
train_y = [1,1,1,0,0,0]

model = LinearSVC()
model.fit(train_x, train_y)

misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

teste_x = [misterio1,misterio2,misterio3]
teste_y = [0,1,1]

previsoes = model.predict(teste_x)
taxa_de_acerto = accuracy_score(teste_y, previsoes)
print('Taxa de acerto %.2f' % (taxa_de_acerto * 100))