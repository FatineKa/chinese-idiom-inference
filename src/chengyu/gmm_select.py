"""gmm_select.py — sélection du nombre de composantes K d'un mélange gaussien
sur des plongements, par la borne du chapitre de sélection de modèle (cas continu
adaptatif : recouvrement + lemme de Massart généralisé + Lipschitz).

Toutes les formules ci-dessous correspondent, ligne à ligne, aux lemmes du chapitre
(voir 0-1-chapitre_selection_modele.html, section 12)."""
import math


def dimension_parametres(K: int, p: int) -> int:
    """d_K : dimension réelle de Theta_K (poids + moyennes + covariances pleines,
    symétriques). Définition 1.7."""
    return (K - 1) + K * p + K * p * (p + 1) // 2


def bornes_g(R: float, sigma_min2: float, sigma_max2: float, p: int):
    """[g_min, g_max] du Lemme 1.6 (bornitude de la perte du mélange)."""
    g_min = (p / 2) * math.log(2 * math.pi * sigma_min2)
    g_max = (p / 2) * math.log(2 * math.pi * sigma_max2) + 2 * R ** 2 / sigma_min2
    return g_min, g_max


def rayon_bloc_max(R: float, sigma_max2: float, p: int) -> float:
    """max(rayon du bloc tau, rayon d'un bloc mu_i, rayon d'un bloc Sigma_i) --
    Theta_K est un produit de 2K+1 blocs (1 simplexe, K boules, K ellipsoïdes de
    matrices), chacun de rayon <= ce maximum. Lemme 1.9 (recouvrement de produit)."""
    return max(1.0, R, math.sqrt(p) * sigma_max2)


def log_recouvrement(K: int, p: int, R: float, sigma_max2: float, eps: float) -> float:
    """log N(eps, Theta_K) -- Lemme 1.9. Theta_K est un produit de 2K+1 blocs ; on
    couvre chaque bloc i à l'échelle eps/sqrt(2K+1) (répartition en norme au carré)
    et on prend le produit des recouvrements (borne de produit), puis on majore
    chaque rayon de bloc par leur maximum commun pour une formule unique."""
    d = dimension_parametres(K, p)
    rho_max = rayon_bloc_max(R, sigma_max2, p)
    n_blocs = 2 * K + 1
    return d * math.log(1 + 2 * rho_max * math.sqrt(n_blocs) / eps)


def log_constante_lipschitz(K: int, p: int, R: float, sigma_min2: float, sigma_max2: float) -> float:
    """log(L_K) — Lemme 1.10 : borne sur |g_theta(x)-g_theta'(x)| <= L_K*||theta-theta'||.
    En log car D_max/D_min = exp(g_max-g_min) explose vite (petit sigma_min, grand R) ;
    voir Remarque 1.20bis sur le caractère très lâche de cette borne générique."""
    g_min, g_max = bornes_g(R, sigma_min2, sigma_max2, p)
    log_ratio = g_max - g_min  # log(D_max/D_min)
    log_L_tau = log_ratio
    log_L_mu = log_ratio + math.log(2 * R / sigma_min2)
    C_sigma = 1 / (2 * sigma_min2) + (2 * R / sigma_min2) ** 2 / 2
    log_L_Sigma = log_ratio + math.log(C_sigma)
    d_tau, d_mu, d_Sigma = 1, p, p * (p + 1) // 2
    # log(sqrt(d_tau*L_tau^2 + d_mu*L_mu^2 + d_Sigma*L_Sigma^2)) via log-sum-exp
    termes = [math.log(d_tau) + 2 * log_L_tau, math.log(d_mu) + 2 * log_L_mu,
              math.log(d_Sigma) + 2 * log_L_Sigma]
    m_ = max(termes)
    log_somme = m_ + math.log(sum(math.exp(t - m_) for t in termes))
    return 0.5 * math.log(K) + 0.5 * log_somme


def lambda_K(K: int, p: int, R: float, sigma_min2: float, sigma_max2: float, m: int, eps: float | None = None):
    """lambda(K) du Théorème 1.11 (Massart généralisé par recouvrement) : le pendant,
    pour ce chapitre, de lambda(K) en Définition 1.11 du chapitre discret — mais
    ici réellement dépendant de K via d_K. Renvoie log(lambda) quand la valeur brute
    dépasserait la plage d'un float64 (cas fréquent ici, voir section 12.5)."""
    if eps is None:
        eps = 1.0 / m
    g_min, g_max = bornes_g(R, sigma_min2, sigma_max2, p)
    b = g_max - g_min
    logN = log_recouvrement(K, p, R, sigma_max2, eps)
    log_L = log_constante_lipschitz(K, p, R, sigma_min2, sigma_max2)
    log_rademacher = math.log(b) + 0.5 * (math.log(2) + math.log(logN) - math.log(m)) if logN > 0 else -math.inf
    log_lipschitz_term = log_L + math.log(eps)
    log_lam = max(log_rademacher, log_lipschitz_term) + math.log1p(
        math.exp(-abs(log_rademacher - log_lipschitz_term)))
    out = {
        "d_K": dimension_parametres(K, p), "b": b, "logN": logN, "log_L": log_L,
        "eps": eps, "log_lambda": log_lam,
    }
    out["lambda"] = math.exp(log_lam) if log_lam < 690 else math.inf
    return out


def epsilon_optimal(K: int, p: int, R: float, sigma_min2: float, sigma_max2: float) -> float:
    """eps* = b/(4*L_K) -- le point d'equilibre entre les deux termes du Theoreme
    1.12 (b*sqrt(2logN(eps)/m) et L_K*eps), au lieu du choix arbitraire eps=1/m
    utilise par lambda_K. Corollaire 1.22 du chapitre (section 12.1) : appeler
    lambda_K(..., eps=epsilon_optimal(...)) resout deja la plus grande partie de
    la mollesse de la borne (section 11.5), sans aucun nouvel outil -- eps=1/m
    n'a jamais ete plus qu'une commodite de calcul, pas un choix optimise."""
    g_min, g_max = bornes_g(R, sigma_min2, sigma_max2, p)
    b = g_max - g_min
    log_L = log_constante_lipschitz(K, p, R, sigma_min2, sigma_max2)
    if log_L >= 690:
        return b / 4  # L_K non representable ; degenere vers la borne triviale
    L_K = math.exp(log_L)
    return b / (4 * L_K) if L_K > 0 else b / 4


def lambda_K_chainage(K: int, p: int, R: float, sigma_min2: float, sigma_max2: float,
                       m: int, n_niveaux: int = 30, base: float = 2.0):
    """Theoreme 1.22 (chapitre, section 12.2) -- chainage a echelles multiples de
    Dudley, sous eps* = epsilon_optimal(...). Sous eps*, la largeur de chaque
    fonction 'pont' (4*L_K*eps_{k-1}) est deja <= b par construction : partir
    du diametre de Theta_K plutot que de eps* redemanderait la borne triviale b a
    chaque niveau grossier avant d'atteindre eps* -- un piege identifie en
    ecrivant ce chapitre (section 12.2, note historique). Sous eps*, la somme
    decroit geometriquement (le terme domine par le niveau de base, raffine par
    un facteur constant, pas par un ordre de grandeur : voir section 12.4)."""
    d_K = dimension_parametres(K, p)
    g_min, g_max = bornes_g(R, sigma_min2, sigma_max2, p)
    b = g_max - g_min
    rho_max = rayon_bloc_max(R, sigma_max2, p)
    n_blocs = 2 * K + 1
    log_L = log_constante_lipschitz(K, p, R, sigma_min2, sigma_max2)

    def logN(eps):
        return d_K * math.log(1 + 2 * rho_max * math.sqrt(n_blocs) / eps)

    if log_L >= 690:
        return {"d_K": d_K, "b": b, "L_K": math.inf, "eps_etoile": None,
                "n_niveaux": 0, "eps_final": None, "degenere": True, "lambda": b}

    L_K = math.exp(log_L)
    eps_etoile = b / (4 * L_K) if L_K > 0 else b / 4
    terme_base = b * math.sqrt(2 * logN(eps_etoile) / m)

    somme = 0.0
    eps_prec = eps_etoile
    for _ in range(n_niveaux):
        eps = eps_prec / base
        largeur = 4 * L_K * eps_prec  # <= b sous eps* (voir docstring) : pas de plafond a appliquer
        # cardinalite exacte des ponts <= N(eps) : chaque point fin a un unique
        # parent (plus proche voisin) dans le filet grossier -- pas N(eps)*N(eps_prec)
        # (une union naive sur les deux filets, plus lache d'un facteur sqrt(2))
        somme += largeur * math.sqrt(2 * logN(eps) / m)
        eps_prec = eps

    residu = L_K * eps_prec  # negligeable des que n_niveaux fait descendre eps_prec << eps_etoile
    borne = terme_base + somme + residu
    return {"d_K": d_K, "b": b, "L_K": L_K, "eps_etoile": eps_etoile,
            "n_niveaux": n_niveaux, "eps_final": eps_prec, "degenere": False,
            "terme_base": terme_base, "somme_chainage": somme, "residu": residu,
            "lambda": borne}


def penalite_birge_massart(K: int, p: int, m: int, kappa: float = 1.0, x_K: float = 1.0):
    """Penalite de type Birge-Massart (2001 ; Massart 2007, St-Flour, Th. 7.11 --
    admis, voir chapitre section 13.1) : pen(K) = kappa * (d_K/m) * (1+sqrt(2*x_K))^2,
    avec sum_K exp(-x_K*d_K) < infini (condition de Kraft, ici triviale car d_K
    croit au moins lineairement en K -- x_K=1 constant suffit).

    Mecanisme different de lambda_K/lambda_K_chainage : la machinerie de
    Birge-Massart travaille sur l'entropie LOCALE en distance de Hellinger, une
    distance toujours dans [0,1] entre densites, plutot que sur un recouvrement en
    norme sup (d'ou vient le facteur D_max/D_min, Lemme 1.11). Elle evite ainsi
    l'explosion exponentielle, mais borne un risque de Hellinger, pas
    R(K)=E[g_theta(X)] utilise ailleurs dans ce chapitre -- non directement
    comparable a lambda(K) sans un pont supplementaire (section 13.4, probleme
    ouvert). kappa n'est pas calcule ici : constante universelle de la theorie,
    non explicitee par Massart (2007) sous une forme immediatement numerique."""
    d_K = dimension_parametres(K, p)
    pen = kappa * (d_K / m) * (1 + math.sqrt(2 * x_K)) ** 2
    return {"d_K": d_K, "x_K": x_K, "kappa": kappa, "penalite": pen}
