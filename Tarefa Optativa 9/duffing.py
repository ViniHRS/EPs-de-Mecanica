#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSCILADOR DE DUFFING - integração numérica pelo método de Runge-Kutta de 4a ordem
==================================================================================

Sistema massa-mola-amortecedor com mola não linear:

    m x'' + c x' + k1 x + k2 x^3 = A cos(w t)

Dividindo por m:

    x'' + delta x' + alpha x + beta x^3 = gamma cos(w t)

    delta = c/m,  alpha = k1/m,  beta = k2/m,  gamma = A/m

Redução a um sistema de 1a ordem (y = [x, v], v = x'):

    x' = v
    v' = gamma cos(w t) - delta v - alpha x - beta x^3

DADOS DE ENTRADA : m, k1, k2, c, A, w, t_max, x(0), x'(0)
DADOS DE SAÍDA   : gráficos x(t), x'(t) e x'(x) (espaço de fases)

USO
---
    python duffing_rk4.py                 -> executa todos os casos da tarefa
    python duffing_rk4.py --personalizado --m 1 --k1 -1 --k2 1 --c 0.3 \
                          --A 0.37 --omega 1.2 --tmax 200 --x0 1 --v0 0
    python duffing_rk4.py --sem-tela      -> apenas grava as figuras (sem janelas)

As figuras são gravadas na MESMA pasta em que este arquivo se encontra.
Bibliotecas necessárias: numpy e matplotlib.
"""

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import matplotlib

if "--sem-tela" in sys.argv:          # permite rodar sem ambiente gráfico
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pasta onde está este arquivo: todas as imagens são gravadas aqui.
try:
    PASTA = Path(__file__).resolve().parent
except NameError:                     # (ex.: execução em um notebook)
    PASTA = Path.cwd()


# =============================================================================
# 1) DADOS DE ENTRADA
# =============================================================================
@dataclass(frozen=True)
class Entrada:
    """Conjunto completo de dados de entrada (itens a-g do enunciado)."""
    nome: str          # identificador do caso (usado nos títulos)
    m: float           # (a) massa do bloco
    k1: float          # (b) constante linear da mola
    k2: float          # (b) constante cúbica da mola
    c: float           # (c) constante de amortecimento viscoso
    A: float           # (d) amplitude da força oscilatória
    omega: float       # (e) frequência angular da força
    tmax: float        # (f) tempo máximo de observação
    x0: float          # (g) posição inicial
    v0: float          # (g) velocidade inicial

    # parâmetros reduzidos da equação dividida por m
    @property
    def delta(self): return self.c / self.m

    @property
    def alpha(self): return self.k1 / self.m

    @property
    def beta(self): return self.k2 / self.m

    @property
    def gamma(self): return self.A / self.m


# =============================================================================
# 2) INTEGRADOR RUNGE-KUTTA DE 4a ORDEM (passo fixo)
# =============================================================================
def rk4_duffing(e: Entrada, h: float = 0.005):
    """
    Integra o oscilador de Duffing em [0, tmax] com passo (aproximadamente) h.

    Retorna os arrays t, x, v (com n+1 pontos, n = tmax/h arredondado).
    O passo é ajustado levemente para que n*h = tmax exatamente.
    """
    delta, alpha, beta, gamma, w = e.delta, e.alpha, e.beta, e.gamma, e.omega

    n = max(1, int(round(e.tmax / h)))
    h = e.tmax / n

    t = np.arange(n + 1) * h          # t_i = i*h (sem acúmulo de erro de soma)
    x = np.empty(n + 1)
    v = np.empty(n + 1)
    x[0], v[0] = e.x0, e.v0

    cos = math.cos

    def acel(tt, xx, vv):
        """v' = x'' = gamma cos(w t) - delta v - alpha x - beta x^3"""
        return gamma * cos(w * tt) - delta * vv - alpha * xx - beta * xx ** 3

    xi, vi = e.x0, e.v0
    for i in range(n):
        ti = t[i]
        # y' = f(t, y), com y = (x, v):  f = (v, acel)
        k1x = vi
        k1v = acel(ti, xi, vi)

        k2x = vi + 0.5 * h * k1v
        k2v = acel(ti + 0.5 * h, xi + 0.5 * h * k1x, vi + 0.5 * h * k1v)

        k3x = vi + 0.5 * h * k2v
        k3v = acel(ti + 0.5 * h, xi + 0.5 * h * k2x, vi + 0.5 * h * k2v)

        k4x = vi + h * k3v
        k4v = acel(ti + h, xi + h * k3x, vi + h * k3v)

        xi += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        vi += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        x[i + 1], v[i + 1] = xi, vi

    return t, x, v


def energia(e: Entrada, x, v):
    """Energia mecânica por unidade de massa: v^2/2 + alpha x^2/2 + beta x^4/4."""
    return 0.5 * v ** 2 + 0.5 * e.alpha * x ** 2 + 0.25 * e.beta * x ** 4


# =============================================================================
# 3) VERIFICAÇÃO DO INTEGRADOR
# =============================================================================
def verificar_ordem_rk4(e: Entrada, tmax_teste=10.0):
    """
    Confere numericamente que o método é de 4a ordem: ao dividir h por 2, o
    erro global deve cair por um fator ~ 2^4 = 16.
    (Usa-se um intervalo curto: em regime caótico o erro cresce exponencialmente.)
    """
    et = replace(e, tmax=tmax_teste)
    ref = rk4_duffing(et, h=0.00125)
    ref_final = np.array([ref[1][-1], ref[2][-1]])
    erros = []
    hs = [0.02, 0.01, 0.005]
    for h in hs:
        _, x, v = rk4_duffing(et, h=h)
        erros.append(np.hypot(x[-1] - ref_final[0], v[-1] - ref_final[1]))
    print("\n[Verificação da ordem do método RK4]  (t_max = %.1f)" % tmax_teste)
    for h, er in zip(hs, erros):
        print(f"   h = {h:<7} erro em t_max = {er:.3e}")
    for i in range(1, len(hs)):
        print(f"   razão erro(h={hs[i-1]})/erro(h={hs[i]}) = {erros[i-1]/erros[i]:.1f}"
              "   (esperado ~ 16)")


# =============================================================================
# 4) GRÁFICOS DE SAÍDA (itens a, b, c)
# =============================================================================
def _marcar_equilibrios(ax_fase, e: Entrada):
    """Marca no espaço de fases os pontos de equilíbrio do sistema autônomo."""
    if e.alpha < 0 and e.beta > 0:                # poço duplo
        xe = math.sqrt(-e.alpha / e.beta)
        ax_fase.plot([-xe, xe], [0, 0], "k*", ms=11, zorder=5,
                     label="equilíbrios estáveis (±√(-α/β))")
        ax_fase.plot([0], [0], "kx", ms=9, mew=2, zorder=5,
                     label="equilíbrio instável (0)")


def figura_caso(e: Entrada, t, x, v, arquivo):
    """Painel com x(t), x'(t) e x'(x)."""
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    ax_x = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[1, 0], sharex=ax_x)
    ax_f = fig.add_subplot(gs[:, 1])

    ax_x.plot(t, x, lw=0.9, color="tab:blue")
    ax_x.set_ylabel("x(t)")
    ax_x.set_title("(a) posição x(t)")
    ax_x.grid(alpha=0.3)

    ax_v.plot(t, v, lw=0.9, color="tab:red")
    ax_v.set_ylabel("ẋ(t)")
    ax_v.set_xlabel("t")
    ax_v.set_title("(b) velocidade ẋ(t)")
    ax_v.grid(alpha=0.3)

    ax_f.plot(x, v, lw=0.6, color="tab:purple", alpha=0.85)
    ax_f.plot(x[0], v[0], "go", ms=9, zorder=6, label="condição inicial")
    ax_f.plot(x[-1], v[-1], "rs", ms=8, zorder=6, label="estado em t_max")
    _marcar_equilibrios(ax_f, e)
    ax_f.set_xlabel("x")
    ax_f.set_ylabel("ẋ")
    ax_f.set_title("(c) espaço de fases ẋ(x)")
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"{e.nome}\n"
        f"m={e.m:g}, k1={e.k1:g}, k2={e.k2:g}, c={e.c:g}, A={e.A:g}, ω={e.omega:g}, "
        f"x(0)={e.x0:g}, ẋ(0)={e.v0:g}   →   "
        f"δ={e.delta:g}, α={e.alpha:g}, β={e.beta:g}, γ={e.gamma:g}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PASTA / arquivo, dpi=150)
    print(f"   figura gravada: {PASTA / arquivo}")
    return fig


def resumo(e: Entrada, t, x, v):
    """Estatísticas simples que auxiliam a descrição do espaço de fases."""
    cruzamentos = int(np.count_nonzero(np.diff(np.sign(x)) != 0))
    print(f"   x ∈ [{x.min():+.3f}, {x.max():+.3f}]   ẋ ∈ [{v.min():+.3f}, {v.max():+.3f}]")
    print(f"   número de passagens de x por zero (trocas de poço): {cruzamentos}")
    print(f"   estado final: x = {x[-1]:+.5f}, ẋ = {v[-1]:+.5f}")
    if e.A == 0:
        E = energia(e, x, v)
        print(f"   energia/m: inicial = {E[0]:+.4f}, final = {E[-1]:+.4f}; "
              f"maior acréscimo num passo = {np.max(np.diff(E)):.2e}  "
              "(~ 0: só há dissipação)")


def executar_caso(e: Entrada, arquivo, h=0.005):
    print(f"\n=== {e.nome} ===")
    print(f"   δ = {e.delta:g}, α = {e.alpha:g}, β = {e.beta:g}, γ = {e.gamma:g}, ω = {e.omega:g}")
    t, x, v = rk4_duffing(e, h)
    resumo(e, t, x, v)
    figura_caso(e, t, x, v, arquivo)
    return t, x, v


# =============================================================================
# 5) SENSIBILIDADE ÀS CONDIÇÕES INICIAIS (item 3 dos detalhes)
# =============================================================================
def _classe_final(base: Entrada, x0, ref_x, h):
    """Verdadeiro se a trajetória que parte de x0 termina 'no mesmo estado' de ref_x."""
    _, x, _ = rk4_duffing(replace(base, x0=x0), h)
    return abs(x[-1] - ref_x) < 0.05


def encontrar_fronteira(base: Entrada, x_ini, x_fim, h, n_varredura=41, iteracoes=45):
    """
    Procura, entre x_ini e x_fim, duas condições iniciais x0 EXTREMAMENTE próximas
    cujas trajetórias terminam (em t_max) em estados diferentes.

    1) varredura grosseira até achar dois pontos vizinhos com desfechos distintos;
    2) bisseção: mantém um extremo com o desfecho 'a' e outro com desfecho diferente,
       reduzindo a distância entre eles pela metade a cada passo.
    """
    grade = np.linspace(x_ini, x_fim, n_varredura)
    finais = [rk4_duffing(replace(base, x0=xi), h)[1][-1] for xi in grade]
    for i in range(len(grade) - 1):
        if abs(finais[i] - finais[i + 1]) > 0.05:
            a, b, ref = grade[i], grade[i + 1], finais[i]
            break
    else:
        return None
    for _ in range(iteracoes):
        m = 0.5 * (a + b)
        if _classe_final(base, m, ref, h):
            a = m
        else:
            b = m
    return a, b


def sensibilidade(base: Entrada, pares, controle: Entrada, eps_controle,
                  arquivo, titulo, h=0.005, ultimos=6000):
    """
    Para cada par (rótulo, x0_a, x0_b) integra as duas trajetórias e mede a distância
    no espaço de fases d(t) = sqrt(Δx² + Δẋ²). Para comparação, faz o mesmo no
    caso de 'controle' (autônomo, A = 0), onde NÃO há sensibilidade.
    O par do meio da lista é o destacado nos painéis (a) e (c).
    """
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_x = fig.add_subplot(gs[0, :])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    cores = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    destaque = len(pares) // 2

    print(f"\n=== SENSIBILIDADE ÀS CONDIÇÕES INICIAIS - {titulo} ===")
    print("   |Δx(0)|          tempo até d > 0,1    λ estimado     x(t_max) a / b")
    for k, (rot, xa0, xb0) in enumerate(pares):
        t, xa, va = rk4_duffing(replace(base, x0=xa0), h)
        _, xb, vb = rk4_duffing(replace(base, x0=xb0), h)
        d = np.hypot(xa - xb, va - vb)
        eps = abs(xb0 - xa0)
        ax_d.semilogy(t, np.maximum(d, 1e-16), color=cores[k % 4], lw=1.1,
                      label=f"{rot}  (|Δx(0)| = {eps:.0e})")

        idx = np.flatnonzero(d > 0.1)
        t_div = t[idx[0]] if idx.size else float("nan")

        # expoente de crescimento exponencial (estimativa grosseira): ajuste de
        # ln d(t) na fase de crescimento, antes da saturação.
        mascara = (d > 10 * eps) & (d < 0.05)
        if np.count_nonzero(mascara) > 20 and idx.size:
            lam = np.polyfit(t[mascara], np.log(d[mascara]), 1)[0]
            txt = f"{lam:+.3f}"
        else:
            txt = "  --  "
        print(f"   {eps:<14.1e}   {t_div:>10.2f} s      {txt:>8}     "
              f"{xa[-1]:+.4f} / {xb[-1]:+.4f}")

        if k == destaque:
            ax_x.plot(t, xa, color="k", lw=1.1, label=f"x(0) = {xa0:.12g}")
            ax_x.plot(t, xb, color=cores[k % 4], lw=1.0, ls="--", alpha=0.9,
                      label=f"x(0) = {xb0:.12g}")
            ax_f.plot(xa[-ultimos:], va[-ultimos:], color="k", lw=0.7, label="trajetória a")
            ax_f.plot(xb[-ultimos:], vb[-ultimos:], color=cores[k % 4], lw=0.7, alpha=0.8,
                      label="trajetória b")

    # controle: sistema autônomo (A = 0) com perturbação do mesmo tipo
    tc, xc1, vc1 = rk4_duffing(controle, h)
    _, xc2, vc2 = rk4_duffing(replace(controle, x0=controle.x0 + eps_controle), h)
    dc = np.hypot(xc1 - xc2, vc1 - vc2)
    ax_d.semilogy(tc, np.maximum(dc, 1e-16), color="gray", lw=1.6, ls="-.",
                  label=f"controle: A = 0  (|Δx(0)| = {eps_controle:.0e})")
    print(f"   controle (A=0): d(0) = {dc[0]:.1e}  →  d(t_max) = {dc[-1]:.1e}"
          "   (as trajetórias CONVERGEM: sem sensibilidade)")

    ax_x.set_title("(a) x(t): duas condições iniciais praticamente idênticas")
    ax_x.set_xlabel("t"); ax_x.set_ylabel("x(t)"); ax_x.grid(alpha=0.3)
    ax_x.legend(fontsize=9, loc="upper left")

    ax_d.set_title("(b) distância no espaço de fases entre as trajetórias")
    ax_d.set_xlabel("t"); ax_d.set_ylabel("d(t) = √(Δx² + Δẋ²)   [escala log]")
    ax_d.grid(alpha=0.3, which="both"); ax_d.legend(fontsize=7.5, loc="lower right")
    ax_d.set_ylim(1e-16, 10)

    ax_f.set_title("(c) espaço de fases (trecho final)")
    ax_f.set_xlabel("x"); ax_f.set_ylabel("ẋ"); ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=8)

    fig.suptitle(f"Sensibilidade às condições iniciais - {titulo}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PASTA / arquivo, dpi=150)
    print(f"   figura gravada: {PASTA / arquivo}")
    return fig


# =============================================================================
# 6) SEÇÃO DE POINCARÉ (auxílio à análise do espaço de fases)
# =============================================================================
def secao_poincare(e: Entrada, n_periodos=3000, descarte=200, passos_por_periodo=200):
    """
    Amostra (x, ẋ) estroboscopicamente em t = n·T, T = 2π/ω (descartando o transiente).
    Movimento periódico de período k·T -> k pontos; caos -> conjunto de estrutura fractal.
    """
    T = 2 * math.pi / e.omega
    h = T / passos_por_periodo
    _, x, v = rk4_duffing(replace(e, tmax=n_periodos * T), h)
    return x[::passos_por_periodo][descarte:], v[::passos_por_periodo][descarte:]


def figura_poincare(casos, arquivo):
    fig, axs = plt.subplots(1, len(casos), figsize=(6.5 * len(casos), 5.8))
    axs = np.atleast_1d(axs)
    print("\n=== SEÇÕES DE POINCARÉ ===")
    for ax, e in zip(axs, casos):
        xs, vs = secao_poincare(e)
        n_dist = len(set(zip(np.round(xs, 3), np.round(vs, 3))))
        print(f"   {e.nome}: {n_dist} ponto(s) distinto(s) em {len(xs)} amostras")
        ax.plot(xs, vs, ".", ms=1.5, color="tab:purple")
        ax.set_xlabel("x(nT)"); ax.set_ylabel("ẋ(nT)"); ax.grid(alpha=0.3)
        ax.set_title(f"{e.nome}\n{n_dist} ponto(s) distinto(s)", fontsize=9)
    fig.suptitle("Seção de Poincaré estroboscópica (t = nT, T = 2π/ω)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PASTA / arquivo, dpi=150)
    print(f"   figura gravada: {PASTA / arquivo}")
    return fig


# =============================================================================
# 7) CONJUNTOS DE DADOS DA TAREFA
# =============================================================================
# Escolhe-se m = 2 para evidenciar a divisão por m; as demais constantes são tais
# que δ = 0,3, α = -1, β = 1 e γ = 0,37 (ω = 1,2), conforme o enunciado.
M = 2.0
CASO_A = Entrada("Caso A (obrigatório) - forçado: δ=0,3; α=-1; β=1; γ=0,37; ω=1,2",
                 m=M, k1=-1.0 * M, k2=1.0 * M, c=0.3 * M, A=0.37 * M, omega=1.2,
                 tmax=200.0, x0=1.0, v0=0.0)

CASO_B = replace(CASO_A, nome="Caso B (obrigatório) - autônomo: mesmos m, k1, k2, c e A = 0",
                 A=0.0, tmax=80.0, x0=0.5, v0=0.0)

CASO_B2 = replace(CASO_B, nome="Caso 3 - autônomo (A = 0) com x(0) = -0,5 e ẋ(0) = 0,3",
                  x0=-0.5, v0=0.3)

CASO_PERIODICO = replace(CASO_A, nome="Caso 4 - forçamento fraco: γ = 0,20 (A = 0,4)",
                         A=0.20 * M, tmax=150.0)

CASO_CAOTICO = replace(CASO_A, nome="Caso 5 - forçamento maior: γ = 0,50 (A = 1,0)",
                       A=0.50 * M)

CASOS = [
    (CASO_A, "duffing_caso1_A_forcado.png"),
    (CASO_B, "duffing_caso2_B_autonomo.png"),
    (CASO_B2, "duffing_caso3_autonomo_outra_condicao.png"),
    (CASO_PERIODICO, "duffing_caso4_gamma_020.png"),
    (CASO_CAOTICO, "duffing_caso5_gamma_050.png"),
]


# =============================================================================
# 8) PROGRAMA PRINCIPAL
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Oscilador de Duffing - integração por RK4")
    p.add_argument("--personalizado", action="store_true",
                   help="executa apenas um caso com os dados fornecidos abaixo")
    p.add_argument("--m", type=float, default=CASO_A.m)
    p.add_argument("--k1", type=float, default=CASO_A.k1)
    p.add_argument("--k2", type=float, default=CASO_A.k2)
    p.add_argument("--c", type=float, default=CASO_A.c)
    p.add_argument("--A", type=float, default=CASO_A.A)
    p.add_argument("--omega", type=float, default=CASO_A.omega)
    p.add_argument("--tmax", type=float, default=CASO_A.tmax)
    p.add_argument("--x0", type=float, default=CASO_A.x0)
    p.add_argument("--v0", type=float, default=CASO_A.v0)
    p.add_argument("--h", type=float, default=0.005, help="passo de integração (padrão 0,005)")
    p.add_argument("--sem-tela", action="store_true", help="não abre janelas; apenas grava PNGs")
    args = p.parse_args()
    h = args.h

    if args.personalizado:
        if args.m <= 0 or args.tmax <= 0:
            sys.exit("Erro: m e tmax devem ser positivos.")
        e = Entrada("Caso personalizado", args.m, args.k1, args.k2, args.c, args.A,
                    args.omega, args.tmax, args.x0, args.v0)
        executar_caso(e, "duffing_personalizado.png", h)
    else:
        verificar_ordem_rk4(CASO_A)

        for e, arq in CASOS:
            executar_caso(e, arq, h)

        # --- item 3: sensibilidade às condições iniciais -----------------------
        controle = replace(CASO_B, tmax=300.0)

        # (i) caso A: busca-se, por bisseção, duas condições iniciais quase idênticas
        #     que conduzem a desfechos diferentes (fronteira entre bacias de atração)
        base_a = replace(CASO_A, tmax=300.0)
        par = encontrar_fronteira(base_a, x_ini=-0.50, x_fim=0.0, h=h)
        if par is None:
            print("\n(não foi encontrada fronteira entre bacias no intervalo pesquisado)")
        else:
            xa, xb = par
            sensibilidade(base_a, [("caso A, par na fronteira", xa, xb)], controle,
                          eps_controle=1e-6, h=h,
                          arquivo="duffing_sensibilidade_caso_A.png",
                          titulo="caso A (γ = 0,37)")

        # (ii) caso γ = 0,50: crescimento exponencial das diferenças (caos)
        base_c = replace(CASO_CAOTICO, tmax=300.0)
        pares = [(f"caso γ=0,50", 1.0, 1.0 + eps) for eps in (1e-3, 1e-6, 1e-9)]
        sensibilidade(base_c, pares, controle, eps_controle=1e-6, h=h,
                      arquivo="duffing_sensibilidade_gamma_050.png",
                      titulo="γ = 0,50 (movimento caótico)")

        # --- auxílio ao item 4: seções de Poincaré --------------------------------
        figura_poincare([CASO_A, CASO_CAOTICO], "duffing_poincare.png")

    if not args.sem_tela:
        plt.show()


if __name__ == "__main__":
    main()
