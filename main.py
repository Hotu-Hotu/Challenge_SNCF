import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import networkx as nx

x_train_df = pd.read_csv("data/x_train_final.csv")
y_train_df = pd.read_csv("data/y_train_final.csv")
test_df = pd.read_csv("data/x_test_final.csv")


# prépa données
df = x_train_df.merge(y_train_df, on="Unnamed: 0")
df = df.drop(columns=["Unnamed: 0.1"], errors="ignore")

#label encoding
for col in ["gare"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

"""# transformations date en jour de la semaine
df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day_name()

# Définir tous les jours possibles
jours = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# Transformer en catégorie avec ordre fixe
df["day"] = pd.Categorical(df["day"], categories=jours)

df = pd.get_dummies(df, columns=["day"])
"""

# Ajout colonne gare origine
df = df.sort_values(["train", "arret"])
df["gare_origine"] = df.groupby("train")["gare"].shift(1)

test_df = test_df.sort_values(["train", "arret"])
test_df["gare_origine"] = test_df.groupby("train")["gare"].shift(1)

# ==============================================================
# ======================MOYENNES================================
# ==============================================================

# Moyenne des retards pour le même train
df["moyenne_p"] = df[["p2q0", "p3q0", "p4q0"]].mean(axis=1)
# Moyenne des retards pour le même quai
df["moyenne_q"] = df[["p0q2", "p0q3", "p0q4"]].mean(axis=1)
# Moyenne des retards sur la journée
df["mean_retard_journalier_gare"] = df.groupby("date")["moyenne_q"].transform("mean")

test_df["moyenne_p"] = test_df[["p2q0", "p3q0", "p4q0"]].mean(axis=1)
test_df["moyenne_q"] = test_df[["p0q2", "p0q3", "p0q4"]].mean(axis=1)
test_df["mean_retard_journalier_gare"] = test_df.groupby("date")["moyenne_q"].transform("mean")

# ==============================================================
# ==============================================================
# ==============================================================





# Assure-toi que "date" est bien au format date (sans timestamp)
df["date"] = pd.to_datetime(df["date"]).dt.date
# Compte le nombre de trains par gare et par jour
df["trains_jour_gare"] = df.groupby(["gare", "date"])["train"].transform("nunique")

test_df["date"] = pd.to_datetime(test_df["date"]).dt.date
test_df["trains_jour_gare"] = test_df.groupby(["gare", "date"])["train"].transform("nunique")


# ==============================================================
# ================= GRAPHE =====================================
# ==============================================================

# df = ["train","date","arret","gare"]
G = nx.DiGraph()

# trier correctement
df = df.sort_values(["train", "date", "arret"])

# création du graphe
for (train, date), group in df.groupby(["train", "date"]):
    prev_gare = None
    prev_arret = None 

    for _, row in group.iterrows():
        gare = row["gare"]
        arret = row["arret"]
        G.add_node(gare)
        
        if prev_gare is not None and prev_arret is not None and arret == prev_arret + 1:
            # poids = nombre de trains passant sur cette arête
            if arret == prev_arret + 1: # vérifier que les arrêts sont consécutifs    
                if G.has_edge(prev_gare, gare):
                    G[prev_gare][gare]["weight"] += 1
                else:
                    G.add_edge(prev_gare, gare, weight=1)
        
        prev_gare = gare
        prev_arret = arret

# ==============================================================
# ==============================================================
# ==============================================================


# Décompte nombre de connexions pour chaque gare
df["deg"] = df["gare"].map(dict(G.degree()))
df["in_deg"] = df["gare"].map(dict(G.in_degree()))
df["out_deg"] = df["gare"].map(dict(G.out_degree()))

test_df["deg"] = df["gare"].map(dict(G.degree()))
test_df["in_deg"] = df["gare"].map(dict(G.in_degree()))
test_df["out_deg"] = df["gare"].map(dict(G.out_degree()))


# ==============================================================
# ==============================================================
# ==============================================================


target = "p0q0"
drop_cols = ["Unnamed: 0", "date", "p0q0", "train"]
features = df.columns.drop(drop_cols)
print(df[features].head())

X = df[features]
y = df[target]
X_test = test_df[features]

# split train / val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

# modèle
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
    )

rf_model.fit(X_train, y_train)

# prédictions
y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)
y_test_pred = rf_model.predict(X_test)


# métriques
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)

print("MAE train:", mae_train)
print("MAE val:", mae_val)


# ==============================================================
# =======================PRINT CSV==============================
# ==============================================================

submission = pd.DataFrame({
    "id": test_df["Unnamed: 0"],
    "p0q0": y_test_pred
})

submission.to_csv("submission.csv", index=False)