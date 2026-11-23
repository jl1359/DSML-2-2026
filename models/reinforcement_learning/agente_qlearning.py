"""
Modulo de Aprendizaje por Refuerzo: Agente Q-Learning para Mitigacion Dinamica de Amenazas.
Materia: Machine Learning y Ciencia de los Datos (2026)
Docente: Lic. Patricia Rodriguez Bilbao
Basado en: 1.11 Aprendizaje por Refuerzo (Procesos de Decision de Markov y Ecuacion de Bellman)
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_phiusiil_processed

GUARDADOS_DIR = os.path.join(BASE_DIR, "models", "reinforcement_learning", "guardados")
RESULTADOS_DIR = os.path.join(BASE_DIR, "models", "reinforcement_learning", "resultados")
DOCS_FIGS_DIR = os.path.join(BASE_DIR, "docs", "pre_informe_1", "figuras")

os.makedirs(GUARDADOS_DIR, exist_ok=True)
os.makedirs(RESULTADOS_DIR, exist_ok=True)
os.makedirs(DOCS_FIGS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

class EntornoDefensaURL:
    """
    Entorno MDP de Simulacion de Filtrado de Trafico Web y Mitigacion de Amenazas.
    Estados: Tupla discreta (nivel_longitud, nivel_simbolos, tiene_https) -> (4 x 3 x 2 = 24 estados)
    Acciones:
        0 = Permitir trafico (Allow)
        1 = Bloquear trafico (Block)
        2 = Inspeccion profunda en Sandbox (Isolate)
    """
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.n_muestras = len(self.df)
        self.indice_actual = 0

    def reset(self):
        self.indice_actual = np.random.randint(0, self.n_muestras)
        return self._obtener_estado()

    def _obtener_estado(self):
        fila = self.df.iloc[self.indice_actual]
        
        # Discretizacion del vector de estado
        # 1. Nivel de longitud de URL (4 categorias)
        longitud = fila.get("URLLength", 0)
        if longitud < -0.5:
            s_len = 0  # Muy corta
        elif longitud < 0.2:
            s_len = 1  # Normal
        elif longitud < 1.0:
            s_len = 2  # Larga
        else:
            s_len = 3  # Muy larga / sospechosa

        # 2. Nivel de caracteres especiales / simbolos (3 categorias)
        simbolos = fila.get("CharContinuationRate", 0)
        if simbolos < -0.2:
            s_sym = 0
        elif simbolos < 0.5:
            s_sym = 1
        else:
            s_sym = 2

        # 3. Presencia de HTTPS seguro (2 categorias: 0 o 1)
        https = int(fila.get("IsHTTPS", 0) > 0)

        return (s_len, s_sym, https)

    def paso(self, accion):
        etiqueta_real = int(self.df.iloc[self.indice_actual]["label"])  # 0: Legitimo, 1: Phishing
        
        # Estructura de Recompensas R(s, a)
        if accion == 0:  # PERMITIR (Allow)
            if etiqueta_real == 0:
                recompensa = 10.0   # Trafico legitimo permitido correctamente
            else:
                recompensa = -100.0  # Phishing permitido (Brecha de Seguridad Critica)
        elif accion == 1:  # BLOQUEAR (Block)
            if etiqueta_real == 1:
                recompensa = 15.0   # Phishing bloqueado exitosamente
            else:
                recompensa = -20.0  # Falso Positivo (Interrupcion de navegacion a usuario)
        elif accion == 2:  # SANDBOX (Aislar para inspeccion)
            if etiqueta_real == 1:
                recompensa = 8.0    # Ataque contenido de forma segura con leve latencia
            else:
                recompensa = 2.0    # Verificacion segura de sitio legitimo
        else:
            recompensa = 0.0

        # Avanzar a la siguiente URL en el flujo de red
        self.indice_actual = np.random.randint(0, self.n_muestras)
        siguiente_estado = self._obtener_estado()
        
        return siguiente_estado, recompensa, etiqueta_real

def entrenar_agente_qlearning():
    print("[INFO] Cargando datos procesados para el agente Q-Learning...")
    train_df, test_df = load_phiusiil_processed()

    entorno = EntornoDefensaURL(train_df)

    # 1. Espacio de Estados y Acciones
    # Dimensiones de la Tabla Q: (4 niveles_len, 3 niveles_sym, 2 https, 3 acciones)
    n_estados = (4, 3, 2)
    n_acciones = 3
    Q = np.zeros(n_estados + (n_acciones,))

    # Hiperparametros de Aprendizaje (Segun diapositivas de la Lic. Patricia Rodriguez)
    alpha = 0.1          # Tasa de aprendizaje
    gamma = 0.9          # Factor de descuento
    epsilon = 1.0        # Tasa inicial de exploracion
    epsilon_min = 0.05   # Exploracion minima
    epsilon_decay = 0.9995
    episodios = 10000
    pasos_por_episodio = 30

    print(f"[INFO] Iniciando entrenamiento de Q-Learning ({episodios:,} episodios)...")
    historial_recompensas = []
    brechas_seguridad = []  # Falsos negativos (permitir phishing)
    aciertos = []

    for ep in range(episodios):
        estado = entorno.reset()
        recompensa_acumulada = 0
        brechas_ep = 0
        aciertos_ep = 0

        for _ in range(pasos_por_episodio):
            # Estrategia Epsilon-Greedy
            if np.random.rand() < epsilon:
                accion = np.random.randint(0, n_acciones)  # Exploracion
            else:
                accion = int(np.argmax(Q[estado]))         # Explotacion

            siguiente_estado, recompensa, etiqueta_real = entorno.paso(accion)

            # Ecuacion de Bellman para Q-Learning:
            # Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
            mejor_q_siguiente = np.max(Q[siguiente_estado])
            delta = recompensa + gamma * mejor_q_siguiente - Q[estado][accion]
            Q[estado][accion] += alpha * delta

            # Metricas de ciberseguridad del paso
            if accion == 0 and etiqueta_real == 1:
                brechas_ep += 1
            if (accion == 0 and etiqueta_real == 0) or (accion in [1, 2] and etiqueta_real == 1):
                aciertos_ep += 1

            estado = siguiente_estado
            recompensa_acumulada += recompensa

        # Decaimiento de epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        historial_recompensas.append(recompensa_acumulada)
        brechas_seguridad.append(brechas_ep)
        aciertos.append(aciertos_ep / pasos_por_episodio)

        if (ep + 1) % 2000 == 0:
            rec_media = np.mean(historial_recompensas[-500:])
            print(f"      Episodio {ep+1:5d} / {episodios} | Recompensa Media: {rec_media:6.1f} | Epsilon: {epsilon:.3f}")

    print("[OK] Entrenamiento de Q-Learning completado exitosamente.")

    # 2. Curvas de Aprendizaje y Convergencia
    ventana = 200
    media_movil = pd.Series(historial_recompensas).rolling(window=ventana).mean()
    tasa_acierto_movil = pd.Series(aciertos).rolling(window=ventana).mean() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Grafica 1: Recompensa acumulada
    ax1.plot(historial_recompensas, alpha=0.15, color="#7f8c8d", label="Recompensa Episodio")
    ax1.plot(media_movil, color="#2980b9", lw=2, label=f"Media Movil ({ventana} ep)")
    ax1.set_title("Curva de Convergencia: Recompensa del Agente Q-Learning", fontweight="bold")
    ax1.set_xlabel("Episodio de Entrenamiento")
    ax1.set_ylabel("Recompensa Acumulada")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Grafica 2: Tasa de Decisiones Seguras
    ax2.plot(tasa_acierto_movil, color="#27ae60", lw=2, label="Efectividad en Seguridad (%)")
    ax2.set_title("Evolucion de la Tasa de Decisiones Correctas", fontweight="bold")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Efectividad (%)")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    grafica_path = os.path.join(RESULTADOS_DIR, "15_convergencia_qlearning.png")
    doc_grafica_path = os.path.join(DOCS_FIGS_DIR, "15_convergencia_qlearning.png")
    plt.savefig(grafica_path, dpi=300)
    plt.savefig(doc_grafica_path, dpi=300)
    plt.close()
    print(f"[OK] Grafica de convergencia guardada en: {grafica_path}")

    # 3. Serializar la Politica Aprendida (Tabla Q)
    q_table_path = os.path.join(GUARDADOS_DIR, "q_table_phishing.pkl")
    joblib.dump(Q, q_table_path)
    print(f"[OK] Politica optima (Tabla Q) serializada en: {q_table_path}")

    # 4. Guardar Metricas Finales
    metricas_rl = {
        "algoritmo": "Q-Learning Tabular (Bellman Optimality)",
        "episodios_totales": episodios,
        "alpha_aprendizaje": alpha,
        "gamma_descuento": gamma,
        "epsilon_final": float(epsilon),
        "recompensa_media_inicial": float(np.mean(historial_recompensas[:500])),
        "recompensa_media_final": float(np.mean(historial_recompensas[-500:])),
        "tasa_acierto_final_porcentaje": float(np.mean(aciertos[-500:]) * 100),
        "falsos_negativos_ultimos_500": int(np.sum(brechas_seguridad[-500:]))
    }

    with open(os.path.join(RESULTADOS_DIR, "metricas_qlearning.json"), "w", encoding="utf-8") as f:
        json.dump(metricas_rl, f, indent=4)
    print(f"[OK] Metricas guardadas en: {os.path.join(RESULTADOS_DIR, 'metricas_qlearning.json')}")

    return Q, metricas_rl

if __name__ == "__main__":
    entrenar_agente_qlearning()
