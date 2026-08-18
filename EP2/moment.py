# Programa deve receber dados de um sistema de n forças,
# calcular momento, torque e invaiante escalar, avaliar
# o sistema e, por fim, gerar uma imagem do sistema
# Código feito com ChatGPT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Funções auxiliares #
TOL = 1e-10


def norma(v):
    # Retorna a norma de um vetor
    return np.linalg.norm(v)

def eh_nulo(v):
    # Verifica se um vetor é numericamente nulo
    return norma(v) < TOL

def vetor_str(v):
    # Formata um vetor para exibição
    return f"({v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f})"

def ler_vetor(nome):
    # Lê as três componentes de um vetor
    print(f"\n{nome}")
    x = float(input("  x = "))
    y = float(input("  y = "))
    z = float(input("  z = "))
    return np.array([x, y, z], dtype=float)

def imprimir_tabela(resultados):
    # Exibe os resultados em formato tabular
    print("\n" + "=" * 72)
    print("RESULTADOS")
    print("=" * 72)

    largura1 = 38
    largura2 = 30

    print(f"{'Grandeza':<{largura1}} | {'Valor':>{largura2}}")
    print("-" * (largura1 + largura2 + 3))

    for nome, valor in resultados:
        print(f"{nome:<{largura1}} | {valor:>{largura2}}")

    print("=" * 72)

def configurar_eixos_3d(ax, pontos):
    # Configura limites, proporção e sistema de referência da figura
    pontos = np.asarray(pontos)

    xmin, ymin, zmin = np.min(pontos, axis=0)
    xmax, ymax, zmax = np.max(pontos, axis=0)

    centro = np.array([
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        (zmin + zmax) / 2
    ])

    dimensoes = np.array([
        xmax - xmin,
        ymax - ymin,
        zmax - zmin
    ])

    maior = max(np.max(dimensoes), 1.0)
    margem = 0.35 * maior

    ax.set_xlim(centro[0] - maior / 2 - margem,
                centro[0] + maior / 2 + margem)

    ax.set_ylim(centro[1] - maior / 2 - margem,
                centro[1] + maior / 2 + margem)

    ax.set_zlim(centro[2] - maior / 2 - margem,
                centro[2] + maior / 2 + margem)

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title("Sistema de forças")

def desenhar_paralelepipedo(ax, pontos):
    # Desenha um paralelepípedo representando o corpo rígido
    # As faces são semitransparentes e as arestas são destacadas
    pontos = np.asarray(pontos)

    xmin, ymin, zmin = np.min(pontos, axis=0)
    xmax, ymax, zmax = np.max(pontos, axis=0)

    # Se alguma dimensão for nula, garante-se que o corpo
    # ainda tenha uma dimensão visualmente apreciável
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    dz = max(zmax - zmin, 1.0)

    # Margem ao redor dos pontos de aplicação
    margem = 0.15

    xmin -= margem * dx
    xmax += margem * dx
    ymin -= margem * dy
    ymax += margem * dy
    zmin -= margem * dz
    zmax += margem * dz

    # Vértices do paralelepípedo
    vertices = [
        [xmin, ymin, zmin],  # 0
        [xmax, ymin, zmin],  # 1
        [xmax, ymax, zmin],  # 2
        [xmin, ymax, zmin],  # 3
        [xmin, ymin, zmax],  # 4
        [xmax, ymin, zmax],  # 5
        [xmax, ymax, zmax],  # 6
        [xmin, ymax, zmax]   # 7
    ]

    # Faces
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # inferior
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # superior
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # frontal
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # direita
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # traseira
        [vertices[3], vertices[0], vertices[4], vertices[7]]   # esquerda
    ]

    # Faces semitransparentes
    faces_collection = Poly3DCollection(
        faces,
        alpha=0.15,
        linewidths=1.5,
        edgecolors="black"
    )

    faces_collection.set_facecolor("lightgray")

    ax.add_collection3d(faces_collection)

    # Arestas
    arestas = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for i, j in arestas:

        a = vertices[i]
        b = vertices[j]

        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color="black",
            linewidth=2.0
        )

def desenhar_vetor(ax, ponto, vetor, label, escala=1.0, color=None, linewidth=2.0):
    # Desenha um vetor 3D a partir de 'ponto'
    v = vetor * escala

    kwargs = {
        "length": norma(v),
        "normalize": False,
        "linewidth": linewidth
    }

    if color is not None:
        kwargs["color"] = color

    ax.quiver(
        ponto[0], ponto[1], ponto[2],
        v[0], v[1], v[2],
        **kwargs
    )

    fim = ponto + v

    ax.text(
        fim[0],
        fim[1],
        fim[2],
        f"  {label}",
        fontsize=10
    )

def desenhar_ponto(ax, ponto, label):
    # Desenha um ponto e sua identificação
    ax.scatter(
        ponto[0],
        ponto[1],
        ponto[2],
        s=40
    )

    ax.text(
        ponto[0],
        ponto[1],
        ponto[2],
        f"  {label}",
        fontsize=10
    )

def desenhar_eixo(ax, ponto, direcao, comprimento, label):
    # Desenha uma reta passando por 'ponto' na direção de 'direcao'
    d = direcao / norma(direcao)

    p1 = ponto - comprimento * d
    p2 = ponto + comprimento * d

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        linestyle="--",
        linewidth=1.5,
        label=label
    )

# Leitura dos dados
print("=" * 72)
print("ANÁLISE DE UM SISTEMA DE FORÇAS EM UM CORPO RÍGIDO")
print("=" * 72)

n = int(input("\nNúmero de forças n = "))

while n < 1:
    print("Erro: n deve ser maior ou igual a 1.")
    n = int(input("Número de forças n = "))

# Componentes das forças e pontos de aplicação
forcas = []
pontos = []

print("\n" + "-" * 72)
print("DADOS DAS FORÇAS")
print("-" * 72)

for i in range(n):
    print(f"\nForça F_{i + 1}:")
    Fx = float(input("  F_x = "))
    Fy = float(input("  F_y = "))
    Fz = float(input("  F_z = "))

    print(f"Ponto de aplicação P_{i + 1}:")
    x = float(input("  x = "))
    y = float(input("  y = "))
    z = float(input("  z = "))

    forcas.append(np.array([Fx, Fy, Fz], dtype=float))
    pontos.append(np.array([x, y, z], dtype=float))

forcas = np.array(forcas)
pontos = np.array(pontos)

# Polo Q
Q = ler_vetor("Polo Q")

# Polo A
A = ler_vetor("Polo A")

# Versor u
u = ler_vetor("Versor u")

if eh_nulo(u):
    raise ValueError("O vetor u não pode ser nulo.")

# Como a entrada deve representar um versor, normalizamos
# numericamente para evitar erros de arredondamento.
u = u / norma(u)

# RESULTANTE

# R = soma das forças
R = np.sum(forcas, axis=0)

# MOMENTO RESULTANTE NO POLO Q

M_Q = np.zeros(3)

for i in range(n):
    r_Q = pontos[i] - Q
    M_Q += np.cross(r_Q, forcas[i])


# MOMENTO RESULTANTE NO POLO A

M_A = np.zeros(3)

for i in range(n):
    r_A = pontos[i] - A
    M_A += np.cross(r_A, forcas[i])

# TORQUE NO EIXO Au

# M_Au = M_A . u
torque_Au = np.dot(M_A, u)

# INVARIANTE ESCALAR

# I = M_Q . R
I = np.dot(M_Q, R)

# CLASSIFICAÇÃO DO SISTEMA

R_nulo = eh_nulo(R)
M_Q_nulo = eh_nulo(M_Q)

if R_nulo and M_Q_nulo:
    classificacao = "Sistema nulo"

elif R_nulo and not M_Q_nulo:
    classificacao = "Redutível a um binário"

elif not R_nulo and abs(I) < TOL:
    classificacao = "Redutível a uma única força"

else:
    classificacao = "Redutível a força + binário"

# EIXO CENTRAL E MOMENTO MÍNIMO

possui_eixo_central = (
    not R_nulo
    and abs(I) >= TOL
)

E0 = None
M_E = None
versor_R = None

if possui_eixo_central:

    R2 = np.dot(R, R)
    R_mod = np.sqrt(R2)

    versor_R = R / R_mod

    # Um ponto pertencente ao eixo central:
    #
    # E = Q + (R x M_Q)/|R|²
    #
    E0 = Q + np.cross(R, M_Q) / R2

    # Momento mínimo:
    #
    # M_E = (I / |R|²) R
    M_E = (I / R2) * R

# EXIBIÇÃO DOS RESULTADOS

resultados = [
    ("Resultante R", vetor_str(R)),
    ("|R|", f"{norma(R):.6f}"),
    ("Momento M_Q", vetor_str(M_Q)),
    ("|M_Q|", f"{norma(M_Q):.6f}"),
    ("Momento M_A", vetor_str(M_A)),
    ("|M_A|", f"{norma(M_A):.6f}"),
    ("Versor u", vetor_str(u)),
    ("Torque no eixo Au", f"{torque_Au:.6f}"),
    ("Invariante escalar I", f"{I:.6f}"),
    ("Classificação", classificacao)
]

if possui_eixo_central:
    resultados.extend([
        ("Ponto E_0 do eixo central", vetor_str(E0)),
        ("Momento mínimo M_E", vetor_str(M_E)),
        ("|M_E|", f"{norma(M_E):.6f}"),
        ("Direção do eixo central", vetor_str(versor_R))
    ])

imprimir_tabela(resultados)

# INFORMAÇÕES ADICIONAIS SOBRE O EIXO CENTRAL

if possui_eixo_central:

    print("\nEquação do eixo central:")
    print(
        "E(λ) = Q + (R × M_Q)/|R|² + λ R/|R|"
    )

    print(
        "\nPortanto:"
    )

    print(
        f"E(λ) = {vetor_str(E0)} "
        f"+ λ {vetor_str(versor_R)}"
    )

# PRIMEIRA FIGURA

fig1 = plt.figure(figsize=(11, 8))
ax1 = fig1.add_subplot(111, projection="3d")

todos_pontos = list(pontos)
todos_pontos.append(Q)
todos_pontos.append(A)

if possui_eixo_central:
    todos_pontos.append(E0)

configurar_eixos_3d(ax1, todos_pontos)
desenhar_paralelepipedo(ax1, pontos)


# Pontos de aplicação
for i in range(n):
    desenhar_ponto(ax1, pontos[i], f"P_{i + 1}")

# Polo Q
desenhar_ponto(ax1, Q, "Q")

# Polo A
desenhar_ponto(ax1, A, "A")


# Forças
for i in range(n):

    F = forcas[i]
    P = pontos[i]

    if not eh_nulo(F):
        desenhar_vetor(
            ax1,
            P,
            F,
            f"F_{i + 1}",
            escala=1.0,
            linewidth=2.0
        )

# Eixo Au
pontos_figura = np.array(todos_pontos)
dimensoes = np.max(pontos_figura, axis=0) - np.min(
    pontos_figura, axis=0
)

comprimento_eixo = max(np.max(dimensoes), 1.0)

desenhar_eixo(
    ax1,
    A,
    u,
    comprimento_eixo,
    "Eixo Au"
)

# Resultante
if not R_nulo:

    desenhar_vetor(
        ax1,
        Q,
        R,
        "R",
        escala=1.0,
        linewidth=2.5
    )

# Momento M_Q
if not M_Q_nulo:

    desenhar_vetor(
        ax1,
        Q,
        M_Q,
        "M_Q",
        escala=1.0,
        linewidth=2.5
    )

# Momento M_A
if not eh_nulo(M_A):

    desenhar_vetor(
        ax1,
        A,
        M_A,
        "M_A",
        escala=1.0,
        linewidth=2.5
    )

ax1.legend()
ax1.set_title(
    "Sistema de forças, polos Q e A e eixo Au"
)

plt.tight_layout()
plt.savefig("figura_sistema_forcas.png", dpi=300)
plt.close()

# SEGUNDA FIGURA — EIXO CENTRAL
if possui_eixo_central:

    fig2 = plt.figure(figsize=(11, 8))
    ax2 = fig2.add_subplot(111, projection="3d")

    # Pontos auxiliares para configurar a escala
    comprimento = max(
        np.max(np.ptp(pontos, axis=0)),
        norma(E0 - Q),
        1.0
    )

    d = versor_R

    # Pontos ao longo do eixo central
    E1 = E0 - comprimento * d
    E2 = E0 + comprimento * d

    todos_pontos_2 = list(pontos)
    todos_pontos_2.extend([
        Q,
        E0,
        E1,
        E2
    ])

    configurar_eixos_3d(ax2, todos_pontos_2)
    desenhar_paralelepipedo(ax2, pontos)

    # Pontos de aplicação
    for i in range(n):
        desenhar_ponto(
            ax2,
            pontos[i],
            f"P_{i + 1}"
        )

    # Polo Q
    desenhar_ponto(ax2, Q, "Q")

    # Ponto E_0
    desenhar_ponto(ax2, E0, "E")


    # Forças
    for i in range(n):

        F = forcas[i]
        P = pontos[i]

        if not eh_nulo(F):

            desenhar_vetor(
                ax2,
                P,
                F,
                f"F_{i + 1}",
                escala=1.0,
                linewidth=2.0
            )


    # Eixo central
    desenhar_eixo(
        ax2,
        E0,
        R,
        comprimento,
        "Eixo central"
    )


    # Resultante R localizada sobre o eixo central
    desenhar_vetor(
        ax2,
        E0,
        R,
        "R",
        escala=1.0,
        linewidth=2.5
    )


    # Momento mínimo M_E
    if not eh_nulo(M_E):

        desenhar_vetor(
            ax2,
            E0,
            M_E,
            "M_E",
            escala=1.0,
            linewidth=2.5
        )


    # Momento no polo Q
    if not M_Q_nulo:

        desenhar_vetor(
            ax2,
            Q,
            M_Q,
            "M_Q",
            escala=1.0,
            linewidth=2.5
        )


    ax2.legend()

    ax2.set_title(
        "Eixo central e representação canônica do sistema de forças"
    )

    plt.tight_layout()
    plt.savefig("figura_eixo_central.png", dpi=300)
    plt.close()