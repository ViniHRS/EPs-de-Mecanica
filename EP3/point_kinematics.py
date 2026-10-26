"""
Cinemática de uma formiga em um carrossel

Convenções usadas:
- O centro do carrossel O é fixo e o raio é R.
- O sentido angular positivo é anti-horário visto de cima.
- v(t) >= 0 é o módulo da velocidade de F PARA DENTRO, em direção ao centro.
- A posição inicial é r(0)=R e theta(0)=theta0.
- omega(t) é dada em rad/s; v(t), em m/s.

Para compilar o programa:
1) source .venv/bin/activate  (ativa o ambiente virtual venv)
2) python point_kinematics.py (compila o programa dentro do ambiente virtual)
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.integrate import cumulative_trapezoid, quad


PASTA = Path(__file__).resolve().parent

# Função para ler um valor real
def pedir_float(mensagem: str, padrao: float) -> float:
    entrada = input(f"{mensagem} [{padrao}]: ").strip()
    # Digitar Enter passa o valor do exemplo para a função
    if not entrada:
        return float(padrao)
    try:
        return float(entrada.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Valor inválido para '{mensagem}': {entrada!r}") from exc

# Função para ler um inteiro. Usa também o valor do exemplo ao apertar Enter
def pedir_int(mensagem: str, padrao: int) -> int:
    entrada = input(f"{mensagem} [{padrao}]: ").strip()
    if not entrada:
        return int(padrao)
    try:
        return int(entrada)
    except ValueError as exc:
        raise ValueError(f"Informe um número inteiro em '{mensagem}'.") from exc

# Função para converter uma expressão em caracteres em uma expressão matemática
def interpretar_funcao(texto: str, simbolo_t: sp.Symbol, nome: str):
    """
    Exemplos aceitos: 2, 1 + 0.2*sin(t), 3*t, exp(-t), pi/2.
    A expressão deve depender somente da variável t.
    """
    texto = texto.strip().replace("^", "**")
    locais = {
        "t": simbolo_t,
        "pi": sp.pi,
        "E": sp.E,
        "e": sp.E,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sqrt": sp.sqrt,
        "exp": sp.exp,
        "log": sp.log,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
    }
    try:
        expressao = sp.sympify(texto, locals=locais)
    except (sp.SympifyError, SyntaxError, TypeError) as exc:
        raise ValueError(
            f"Não foi possível interpretar {nome}(t) = {texto!r}. "
            "Digite somente uma expressão em t, por exemplo: 1 + sin(t)."
        ) from exc

    if not isinstance(expressao, sp.Expr):
        raise ValueError(f"A expressão de {nome}(t) não é uma expressão escalar válida.")
    simbolos_desconhecidos = expressao.free_symbols - {simbolo_t}
    if simbolos_desconhecidos:
        nomes = ", ".join(sorted(str(s) for s in simbolos_desconhecidos))
        raise ValueError(f"A função {nome}(t) contém variável(is) não reconhecida(s): {nomes}.")

    derivada = sp.diff(expressao, simbolo_t)
    try:
        funcao = sp.lambdify(simbolo_t, expressao, modules=["numpy"])
        funcao_derivada = sp.lambdify(simbolo_t, derivada, modules=["numpy"])
    except Exception as exc:
        raise ValueError(f"Não foi possível converter a função {nome}(t) para cálculo numérico.") from exc
    return expressao, derivada, funcao, funcao_derivada

# Avalia uma função lambdificada e normaliza o resultado para float/array
def avaliar(funcao: Callable, t: np.ndarray | float) -> np.ndarray | float:
    resultado = np.asarray(funcao(t), dtype=float)
    if np.any(~np.isfinite(resultado)):
        raise ValueError("Uma função fornecida resultou em NaN ou infinito no intervalo escolhido.")
    if np.ndim(t) == 0:
        return float(resultado)
    if resultado.ndim == 0:
        return np.full(np.shape(t), float(resultado), dtype=float)
    return np.broadcast_to(resultado, np.shape(t)).astype(float, copy=False)

# Função para integrar f(t) de 0 até um instante t
def integrar_ate(funcao: Callable, instante: float) -> float:
    if instante == 0:
        return 0.0
    resultado, _ = quad(lambda x: float(avaliar(funcao, x)), 0.0, instante, limit=200)
    return float(resultado)

# Função para calcular r, theta e todas as grandezas vetoriais no intervalo de tempo
def montar_dados(t: np.ndarray, R: float, theta0: float, omega_fn: Callable,
                 alpha_fn: Callable, v_fn: Callable, dv_fn: Callable):
    omega = avaliar(omega_fn, t)
    alpha = avaliar(alpha_fn, t)
    v = avaliar(v_fn, t)
    dv = avaliar(dv_fn, t)

    if np.any(v < -1e-9):
        raise ValueError(
            "A função v(t) assumiu valores negativos. Neste programa, v(t) é o módulo "
            "da velocidade dirigida para o centro e deve ser não negativa."
        )
    v = np.maximum(v, 0.0)

    deslocamento_radial = cumulative_trapezoid(v, t, initial=0.0)
    r = R - deslocamento_radial
    theta = theta0 + cumulative_trapezoid(omega, t, initial=0.0)

    er = np.column_stack((np.cos(theta), np.sin(theta)))
    et = np.column_stack((-np.sin(theta), np.cos(theta)))
    xy = r[:, None] * er

    # Velocidades: v_abs = v_rel + v_arr.
    v_rel = (-v)[:, None] * er
    v_arr = (r * omega)[:, None] * et
    v_abs = v_rel + v_arr

    # Acelerações: a_abs = a_rel + a_arr + a_C.
    a_rel = (-dv)[:, None] * er
    a_arr = (-r * omega**2)[:, None] * er + (r * alpha)[:, None] * et
    a_cor = (-2.0 * omega * v)[:, None] * et
    a_abs = a_rel + a_arr + a_cor

    return {
        "t": t, "r": r, "theta": theta, "x": xy[:, 0], "y": xy[:, 1],
        "omega": omega, "alpha": alpha, "v": v, "dv": dv,
        "v_rel": v_rel, "v_arr": v_arr, "v_abs": v_abs,
        "a_rel": a_rel, "a_arr": a_arr, "a_cor": a_cor, "a_abs": a_abs,
    }

# Função para calcular todas as grandezas em um instante t >= 0
def estado_no_instante(instante: float, R: float, theta0: float,
                       omega_fn: Callable, alpha_fn: Callable,
                       v_fn: Callable, dv_fn: Callable) -> dict:
    omega = float(avaliar(omega_fn, instante))
    alpha = float(avaliar(alpha_fn, instante))
    v = float(avaliar(v_fn, instante))
    dv = float(avaliar(dv_fn, instante))
    if v < -1e-9:
        raise ValueError("v(t) ficou negativa; use uma função de velocidade dirigida para dentro.")
    v = max(v, 0.0)

    r = R - integrar_ate(v_fn, instante)
    if r < -1e-5:
        raise ValueError("Esse instante ocorre depois de F atingir o centro do carrossel.")
    r = max(0.0, r)
    theta = theta0 + integrar_ate(omega_fn, instante)

    er = np.array([math.cos(theta), math.sin(theta)])
    et = np.array([-math.sin(theta), math.cos(theta)])
    pos = r * er
    v_rel = -v * er
    v_arr = r * omega * et
    v_abs = v_rel + v_arr
    a_rel = -dv * er
    a_arr = -r * omega**2 * er + r * alpha * et
    a_cor = -2.0 * omega * v * et
    a_abs = a_rel + a_arr + a_cor

    return {
        "t": instante, "r": r, "theta": theta, "x": pos[0], "y": pos[1],
        "omega": omega, "alpha": alpha, "v": v, "dv": dv,
        "v_rel": v_rel, "v_arr": v_arr, "v_abs": v_abs,
        "a_rel": a_rel, "a_arr": a_arr, "a_cor": a_cor, "a_abs": a_abs,
    }

# Função para gravar grandezas escalares e componentes cartesianas dos vetores em um arquvio csv
def escrever_csv(dados: dict, caminho: Path) -> None:
    campos = ["t_s", "r_m", "theta_rad", "x_m", "y_m", "omega_rad_s", "alpha_rad_s2",
              "v_inward_m_s", "dv_dt_m_s2"]
    for nome in ("v_rel", "v_arr", "v_abs", "a_rel", "a_arr", "a_cor", "a_abs"):
        campos.extend([f"{nome}_x", f"{nome}_y"])

    with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=";")
        escritor.writeheader()
        n = len(dados["t"])
        for i in range(n):
            linha = {
                "t_s": dados["t"][i], "r_m": dados["r"][i],
                "theta_rad": dados["theta"][i], "x_m": dados["x"][i],
                "y_m": dados["y"][i], "omega_rad_s": dados["omega"][i],
                "alpha_rad_s2": dados["alpha"][i], "v_inward_m_s": dados["v"][i],
                "dv_dt_m_s2": dados["dv"][i],
            }
            for nome in ("v_rel", "v_arr", "v_abs", "a_rel", "a_arr", "a_cor", "a_abs"):
                linha[f"{nome}_x"] = dados[nome][i, 0]
                linha[f"{nome}_y"] = dados[nome][i, 1]
            escritor.writerow(linha)


def salvar_trajetoria(dados: dict, R: float, caminho: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Circle((0, 0), R, facecolor="#eeeeee", edgecolor="black", alpha=0.8, zorder=0))
    ax.plot(dados["x"], dados["y"], linewidth=2.2, color="#d62728", label="Trajetória de F")
    ax.scatter([dados["x"][0]], [dados["y"][0]], color="#2ca02c", s=55, zorder=4, label="Posição inicial")
    ax.scatter([dados["x"][-1]], [dados["y"][-1]], color="#d62728", s=55, zorder=4, label="Posição final")
    ax.scatter([0], [0], color="black", s=25, zorder=5, label="Centro O")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.25 * R, 1.25 * R)
    ax.set_ylim(-1.25 * R, 1.25 * R)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajetória de F vista de cima")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=160)
    plt.close(fig)

# Função para gerar o GIF do movimento, alternando a representação das velocidades e das acelerações
def criar_animacao(dados: dict, R: float, delta_t: float, caminho: Path, fps: int = 20):
    t = dados["t"]
    n = len(t)
    if n < 2:
        raise ValueError("Não há pontos suficientes para construir a animação.")

    nomes_v = ("v_rel", "v_arr", "v_abs")
    nomes_a = ("a_rel", "a_arr", "a_cor", "a_abs")
    cores = {
        "v_rel": "#1f77b4", "v_arr": "#ff7f0e", "v_abs": "#2ca02c",
        "a_rel": "#9467bd", "a_arr": "#d62728", "a_cor": "#8c564b", "a_abs": "#e377c2",
    }
    rotulos = {
        "v_rel": r"$v_{rel}$", "v_arr": r"$v_{arr}$", "v_abs": r"$v_{abs}$",
        "a_rel": r"$a_{rel}$", "a_arr": r"$a_{arr}$", "a_cor": r"$a_C$", "a_abs": r"$a_{abs}$",
    }

    def maior_modulo(lista):
        return max(float(np.max(np.linalg.norm(dados[chave], axis=1))) for chave in lista)

    # Escalas constantes por categoria: vetores da mesma categoria são comparáveis ao longo do GIF.
    max_v = max(maior_modulo(nomes_v), 1e-9)
    max_a = max(maior_modulo(nomes_a), 1e-9)
    escala_v = 0.28 * R / max_v
    escala_a = 0.28 * R / max_a

    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(Circle((0, 0), R, facecolor="#efefef", edgecolor="#222222", alpha=0.8, zorder=0))
    # Anel auxiliar para reforçar o contorno físico do carrossel.
    ax.add_patch(Circle((0, 0), 0.96 * R, fill=False, edgecolor="#aaaaaa", linewidth=1.0, zorder=1))
    ax.scatter([0], [0], color="black", s=22, zorder=5)
    centro_texto = ax.text(0.02, 0.02, "O", fontsize=9, ha="left", va="bottom", zorder=6)

    # Raios/elementos gráficos giram junto com o carrossel.
    n_raios = 12
    raios = [ax.plot([], [], color="#9a9a9a", linewidth=1.0, alpha=0.85, zorder=1)[0]
             for _ in range(n_raios)]
    raio_guia, = ax.plot([], [], color="#4c78a8", linewidth=3.0, alpha=0.75, zorder=2)
    trilha, = ax.plot([], [], color="#d62728", linewidth=1.5, alpha=0.7, zorder=3)
    formiga, = ax.plot([], [], marker="o", markersize=9, color="#d62728", markeredgecolor="white",
                       markeredgewidth=1.0, linestyle="None", zorder=8)
    rotulo_F = ax.text(0, 0, " F", fontsize=10, weight="bold", zorder=9)
    texto_tempo = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#cccccc"))
    texto_modo = ax.text(0.98, 0.98, "", transform=ax.transAxes, va="top", ha="right", fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#cccccc"))

    # Um objeto quiver por vetor para permitir atualizar posição e componentes em cada frame.
    quivers = {}
    for nome in (*nomes_v, *nomes_a):
        quivers[nome] = ax.quiver([0.0], [0.0], [0.0], [0.0], angles="xy", scale_units="xy", scale=1,
                                  color=cores[nome], width=0.006, headwidth=4.5, headlength=6.0,
                                  headaxislength=5.0, zorder=10)

    lim = 1.52 * R
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Movimento de F no carrossel — vista de cima", pad=16)
    ax.grid(True, alpha=0.18)

    legenda = [Line2D([0], [0], color=cores[nome], linewidth=2.5, marker=">", markersize=5, label=rotulos[nome])
               for nome in (*nomes_v, *nomes_a)]
    fig.legend(handles=legenda, loc="lower center", ncol=4, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, 0.015))
    fig.subplots_adjust(bottom=0.13, top=0.91, left=0.09, right=0.96)

    def atualizar(i):
        th = dados["theta"][i]
        x = dados["x"][i]
        y = dados["y"][i]
        rr = max(0.0, dados["r"][i])
        for k, linha in enumerate(raios):
            angulo = th + 2.0 * math.pi * k / n_raios
            linha.set_data([0.0, R * math.cos(angulo)], [0.0, R * math.sin(angulo)])
        raio_guia.set_data([0.0, x], [0.0, y])
        trilha.set_data(dados["x"][:i + 1], dados["y"][:i + 1])
        formiga.set_data([x], [y])
        rotulo_F.set_position((x + 0.035 * R, y + 0.035 * R))
        texto_tempo.set_text(f"t = {t[i]:.2f} s\nr = {rr:.2f} m\nθ = {th:.2f} rad")

        fase = int(math.floor((t[i] + 1e-10) / delta_t))
        mostrar_velocidades = (fase % 2 == 0)
        ativos = nomes_v if mostrar_velocidades else nomes_a
        texto_modo.set_text("VELOCIDADES" if mostrar_velocidades else "ACELERAÇÕES")
        escala = escala_v if mostrar_velocidades else escala_a
        for nome, quiver in quivers.items():
            quiver.set_offsets(np.array([[x, y]]))
            if nome in ativos:
                vec = dados[nome][i] * escala
                quiver.set_UVC([vec[0]], [vec[1]])
                quiver.set_visible(True)
            else:
                quiver.set_UVC([0.0], [0.0])
                quiver.set_visible(False)

        return [*raios, raio_guia, trilha, formiga, rotulo_F, texto_tempo, texto_modo, *quivers.values(), centro_texto]

    animacao = FuncAnimation(fig, atualizar, frames=n, interval=1000 / fps, blit=False)
    animacao.save(caminho, writer=PillowWriter(fps=fps), dpi=105)
    # Mostra a animação na tela somente se houver um backend gráfico interativo.
    # Em execução sem interface gráfica, o GIF salvo continua disponível normalmente.
    backend = plt.get_backend().lower()
    interativos = ("qtagg", "qt5agg", "tkagg", "wxagg", "gtk3agg", "macosx", "webagg")
    try:
        if any(nome in backend for nome in interativos):
            plt.show()
    finally:
        plt.close(fig)
    return animacao

# Função para exibir a lei do movimento em forma paramétrica, usando integrais se necessário
def imprimir_leis_simbolicas(R: float, theta0: float, expr_omega: sp.Expr, expr_v: sp.Expr,
                             simbolo_t: sp.Symbol) -> None:
    s = sp.Symbol("s", real=True)
    r_expr = sp.Float(R) - sp.integrate(expr_v.subs(simbolo_t, s), (s, 0, simbolo_t))
    theta_expr = sp.Float(theta0) + sp.integrate(expr_omega.subs(simbolo_t, s), (s, 0, simbolo_t))
    r_expr = sp.simplify(r_expr)
    theta_expr = sp.simplify(theta_expr)
    print("\nLEI DE MOVIMENTO (forma paramétrica)")
    print(f"r(t)     = {sp.sstr(r_expr)}")
    print(f"theta(t) = {sp.sstr(theta_expr)}")
    print(f"x_F(t)   = r(t) cos(theta(t))")
    print(f"y_F(t)   = r(t) sin(theta(t))")
    print("(As integrais são tomadas de 0 até t; se uma integral não tiver forma fechada, ela aparece como Integral.)")

# Função main
def main() -> int:
    print("=" * 70)
    print("Cinemática de uma Formiga em em Carrossel")
    print("=" * 70)
    print("Digite omega(t) em rad/s e v(t) em m/s; v(t) é positiva para dentro.")
    print("Exemplos (valores entendidos como padrão pelo programa):\n" 
          "omega(t) = 1 + 0.2*sin(t)\n"
          "v(t) = 0.5 + 0.1*sin(t)")

    try:
        R = pedir_float("Raio do carrossel R (m)", 5.0)
        if R <= 0:
            raise ValueError("O raio R deve ser positivo.")
        theta0_graus = pedir_float("Ângulo inicial theta0 (graus; anti-horário positivo)", 0.0)
        theta0 = math.radians(theta0_graus)
        entrada_omega = input("Função omega(t) [1 + 0.2*sin(t)]: ").strip() or "1 + 0.2*sin(t)"
        entrada_v = input("Função v(t), positiva para o centro [0.5 + 0.1*sin(t)]: ").strip() or "0.5 + 0.1*sin(t)"
        T = pedir_float("Duração máxima da simulação T (s)", 8.0)
        if T <= 0:
            raise ValueError("A duração T deve ser positiva.")
        delta_t = pedir_float("Intervalo para alternar velocidades/acelerações Δt (s)", 1.0)
        if delta_t <= 0:
            raise ValueError("O intervalo Δt deve ser positivo.")
        numero_frames = pedir_int("Número de quadros da animação (maior = mais suave)", 81)
        if numero_frames < 2:
            raise ValueError("Use pelo menos 2 quadros.")
        fps = pedir_int("Quadros por segundo do GIF", 12)
        if fps < 1:
            raise ValueError("FPS deve ser pelo menos 1.")

        simbolo_t = sp.Symbol("t", real=True)
        expr_omega, expr_alpha, omega_fn, alpha_fn = interpretar_funcao(entrada_omega, simbolo_t, "omega")
        expr_v, expr_dv, v_fn, dv_fn = interpretar_funcao(entrada_v, simbolo_t, "v")

        # Avalia primeiro na malha solicitada e interrompe a simulação quando F chega ao centro.
        t_inicial = np.linspace(0.0, T, numero_frames)
        omega_inicial = avaliar(omega_fn, t_inicial)
        v_inicial = avaliar(v_fn, t_inicial)
        if np.any(v_inicial < -1e-9):
            raise ValueError("v(t) deve ser não negativa no intervalo: F desloca-se do bordo para o centro.")
        r_inicial = R - cumulative_trapezoid(np.maximum(v_inicial, 0.0), t_inicial, initial=0.0)
        atingiu = np.flatnonzero(r_inicial <= 0.0)
        chegou_ao_centro = len(atingiu) > 0
        if chegou_ao_centro:
            idx = int(atingiu[0])
            if idx == 0:
                T_efetivo = 0.0
            else:
                r1, r2 = r_inicial[idx - 1], r_inicial[idx]
                frac = r1 / (r1 - r2) if abs(r1 - r2) > 1e-12 else 1.0
                T_efetivo = float(t_inicial[idx - 1] + frac * (t_inicial[idx] - t_inicial[idx - 1]))
            if T_efetivo <= 0:
                raise ValueError("A formiga já estaria no centro no instante inicial; confira R e v(t).")
            t = np.linspace(0.0, T_efetivo, max(2, int(round(numero_frames * T_efetivo / T))))
            print(f"\nAviso: F atinge o centro antes de T. A simulação será encerrada em t = {T_efetivo:.4f} s.")
        else:
            t = t_inicial

        dados = montar_dados(t, R, theta0, omega_fn, alpha_fn, v_fn, dv_fn)
        if chegou_ao_centro:
            # O instante de chegada foi estimado a partir dos dados amostrados.
            # Impomos a condição geométrica exata no último quadro.
            dados["r"][-1] = 0.0
            dados["x"][-1] = 0.0
            dados["y"][-1] = 0.0
            dados["v_arr"][-1] = np.zeros(2)
            dados["v_abs"][-1] = dados["v_rel"][-1]
            dados["a_arr"][-1] = np.zeros(2)
            dados["a_abs"][-1] = dados["a_rel"][-1] + dados["a_cor"][-1]
        imprimir_leis_simbolicas(R, theta0, expr_omega, expr_v, simbolo_t)

        arquivo_gif = PASTA / "animacao_carrossel.gif"
        arquivo_png = PASTA / "trajetoria.png"
        arquivo_csv = PASTA / "resultados_cinematica.csv"
        salvar_trajetoria(dados, R, arquivo_png)
        escrever_csv(dados, arquivo_csv)
        criar_animacao(dados, R, delta_t, arquivo_gif, fps=fps)

        print("\nArquivos criados na pasta do programa:")
        print(f"  Animação GIF : {arquivo_gif}")
        print(f"  Trajetória PNG: {arquivo_png}")
        print(f"  Resultados CSV: {arquivo_csv}")
        print(f"\nInstante final simulado: {dados['t'][-1]:.4f} s")

        print("\nCONSULTA EM UM INSTANTE t")
        print("Digite um instante entre 0 e o instante final simulado. Enter encerra as consultas.")
        while True:
            entrada = input("t (s) = ").strip()
            if not entrada:
                break
            try:
                instante = float(entrada.replace(",", "."))
                if instante < 0 or instante > dados["t"][-1] + 1e-8:
                    print(f"Digite t entre 0 e {dados['t'][-1]:.4f} s.")
                    continue
                estado = estado_no_instante(instante, R, theta0, omega_fn, alpha_fn, v_fn, dv_fn)
                print(f"\nt = {instante:.4f} s | r = {estado['r']:.6g} m | theta = {estado['theta']:.6g} rad")
                for nome, unidade in (("v_rel", "m/s"), ("v_arr", "m/s"), ("v_abs", "m/s"),
                                      ("a_rel", "m/s²"), ("a_arr", "m/s²"), ("a_cor", "m/s²"),
                                      ("a_abs", "m/s²")):
                    vetor = estado[nome]
                    print(f"  {nome:6s} = ({vetor[0]: .6g}, {vetor[1]: .6g}) {unidade}; módulo = {np.linalg.norm(vetor):.6g} {unidade}")
            except (ValueError, OverflowError, ZeroDivisionError) as exc:
                print(f"Não foi possível calcular nesse instante: {exc}")
        return 0

    except (ValueError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\nExecução interrompida pelo usuário.")
            return 130
        print(f"\nErro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())