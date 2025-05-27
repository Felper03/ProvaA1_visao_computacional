import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class ClassificadorCNN:
    def __init__(self, pasta_treinamento, pasta_classificacao, tamanho=(128,128)):
        self.pasta_treinamento = pasta_treinamento
        self.pasta_classificacao = pasta_classificacao
        self.tamanho = tamanho
        self.class_names = ['gato', 'cachorro']
        self.model = None

    def carregar_treinamento(self):
        imagens = []
        rotulos = []

        print(f"Carregando dados de: {self.pasta_treinamento}")
        for pasta_classe in os.listdir(self.pasta_treinamento):
            caminho_classe = os.path.join(self.pasta_treinamento, pasta_classe)
            if not os.path.isdir(caminho_classe):
                continue

            for arquivo in os.listdir(caminho_classe):
                if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                    caminho_arquivo = os.path.join(caminho_classe, arquivo)
                    img = cv2.imread(caminho_arquivo)
                    if img is None:
                        continue
                    img = cv2.resize(img, self.tamanho)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imagens.append(img)

                    if pasta_classe.lower() == 'gato':
                        rotulos.append(0)
                    elif pasta_classe.lower() == 'cachorro':
                        rotulos.append(1)

        imagens = np.array(imagens, dtype='float32') / 255.0
        rotulos = np.array(rotulos)
        print(f"Total de imagens carregadas: {len(imagens)}")
        return imagens, rotulos

    def criar_modelo(self):
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(*self.tamanho, 3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def treinar(self, epochs=10):
        imagens, rotulos = self.carregar_treinamento()

        X_train, X_test, y_train, y_test = train_test_split(
            imagens, rotulos, test_size=0.2, random_state=42, stratify=rotulos
        )

        self.criar_modelo()
        history = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

        # Avaliação com métricas detalhadas
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        print("\nMétricas de Avaliação no conjunto de teste:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        # Gráfico da acurácia
        plt.figure(figsize=(8,5))
        plt.plot(history.history['accuracy'], label='Treino')
        plt.plot(history.history['val_accuracy'], label='Validação')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.title('Treinamento da CNN')
        plt.show()

    def classificar_imagens(self):
        if self.model is None:
            print("Modelo não treinado ainda.")
            return

        arquivos = [f for f in os.listdir(self.pasta_classificacao) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for arquivo in arquivos:
            caminho = os.path.join(self.pasta_classificacao, arquivo)
            img = cv2.imread(caminho)
            if img is None:
                print(f"Erro ao ler a imagem {arquivo}")
                continue
            img_resized = cv2.resize(img, self.tamanho)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype('float32') / 255.0
            img_exp = np.expand_dims(img_norm, axis=0)

            pred = self.model.predict(img_exp)
            classe_pred = np.argmax(pred)
            nome_classe = self.class_names[classe_pred]
            print(f"A imagem '{arquivo}' foi classificada como: {nome_classe}")


if __name__ == "__main__":
    # Ajuste o caminho para o seu dataset de treino do Kaggle
    pasta_treino = r"C:\Users\FELIPEGABRIELFERREIR\Downloads\archive\train"
    pasta_imagens_para_classificar = "imagens"

    classificador = ClassificadorCNN(pasta_treino, pasta_imagens_para_classificar)
    classificador.treinar(epochs=10)
    classificador.classificar_imagens()
