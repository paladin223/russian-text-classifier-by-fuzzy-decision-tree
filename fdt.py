import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from web_app.app.preprocessing import clean_text

STOP_WORDS = "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между".split()

DEFAULT_MODEL_PATH = Path("fuzzy_text_model_5cls_tuned_12_05_0040.pkl")
DEFAULT_DATASET_PATH = Path("lenta-ru-news.csv")
DEFAULT_TOPICS = ["Спорт", "Экономика", "Культура", "Наука и техника", "Путешествия"]


class FuzzyNode:
    def __init__(self, depth=0):
        self.depth = depth
        self.feature_name = None
        self.feature_idx = None
        self.branches = {}
        self.is_leaf = False
        self.leaf_value = None


class FuzzyDecisionTree:
    def __init__(self, max_depth=22, min_gain=0.0015):
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.root = None

    def _membership(self, values):
        k = 150.0
        thr = 0.03
        val_clipped = np.clip(values, -1, 1)
        mu_high = 1.0 / (1.0 + np.exp(-k * (val_clipped - thr)))
        mu_low = 1.0 - mu_high
        return mu_low, mu_high

    def _entropy(self, y, weights):
        total = np.sum(weights)
        if total < 1e-9:
            return 0
        df = pd.DataFrame({"w": weights, "y": y})
        probs = df.groupby("y")["w"].sum() / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def fit(self, X, y, feature_names):
        print(f"Обучение: перебор {len(feature_names)} признаков...")
        self.feature_names = feature_names
        self.root = self._build_tree(X, y, np.ones(X.shape[0]), 0)

    def _build_tree(self, X, y, weights, depth):
        node = FuzzyNode(depth)
        df_temp = pd.DataFrame({"y": y, "w": weights})
        class_sums = df_temp.groupby("y")["w"].sum()
        total_weight = np.sum(weights)

        if class_sums.empty:
            node.is_leaf = True
            node.leaf_value = "Unknown"
            return node

        node.leaf_value = class_sums.idxmax()
        if depth >= self.max_depth or total_weight < 2.0 or len(class_sums) == 1:
            node.is_leaf = True
            return node

        curr_ent = self._entropy(y, weights)
        best_gain = -1.0
        best_idx = -1
        best_splits = None

        active_cols = np.where(np.sum(X * weights[:, None], axis=0) > 0.01)[0]
        if len(active_cols) == 0:
            node.is_leaf = True
            return node

        for i in active_cols:
            col = X[:, i]
            mu_low, mu_high = self._membership(col)
            w_l, w_h = weights * mu_low, weights * mu_high
            s_l, s_h = np.sum(w_l), np.sum(w_h)
            if s_l < 0.1 or s_h < 0.1:
                continue

            child_ent = (s_l / total_weight) * self._entropy(y, w_l) + (s_h / total_weight) * self._entropy(y, w_h)
            gain = curr_ent - child_ent
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_splits = {"Low": w_l, "High": w_h}

        if best_gain <= self.min_gain or best_idx == -1:
            node.is_leaf = True
            return node

        node.feature_idx = best_idx
        node.feature_name = self.feature_names[best_idx]
        node.branches["Low"] = self._build_tree(X, y, best_splits["Low"], depth + 1)
        node.branches["High"] = self._build_tree(X, y, best_splits["High"], depth + 1)
        return node

    def predict(self, vec):
        votes = {}
        self._traverse(self.root, vec, 1.0, votes)
        if not votes:
            return "Unknown"
        return max(votes, key=votes.get)

    def _traverse(self, node, vec, weight, votes):
        if weight < 0.01:
            return
        if node.is_leaf:
            votes[node.leaf_value] = votes.get(node.leaf_value, 0) + weight
            return

        val = vec[node.feature_idx]
        mu_high = 1.0 / (1.0 + math.exp(-150.0 * (val - 0.03)))
        mu_low = 1.0 - mu_high
        self._traverse(node.branches["Low"], vec, weight * mu_low, votes)
        self._traverse(node.branches["High"], vec, weight * mu_high, votes)


def save_pipeline(model_path, model, tfidf, selector, topics):
    payload = {
        "model": model,
        "tfidf": tfidf,
        "selector": selector,
        "topics": topics,
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)


def load_pipeline(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_for_text(raw_text, pipeline):
    cleaned = clean_text(raw_text)
    x_raw = pipeline["tfidf"].transform([cleaned])
    x_sel = pipeline["selector"].transform(x_raw).toarray()
    return pipeline["model"].predict(x_sel[0])


def sample_topics(df, topics, samples_per_class, random_state):
    frames = []
    for topic in topics:
        subset = df[df["topic"] == topic]
        if subset.empty:
            print(f"[WARN] Класс '{topic}' не найден в выбранных строках датасета.")
            continue
        take_n = min(samples_per_class, len(subset))
        frames.append(subset.sample(take_n, random_state=random_state))
        print(f"   > {topic:<12}: {take_n} примеров")

    if not frames:
        raise ValueError("Не удалось собрать ни одного класса для обучения.")
    return pd.concat(frames).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def train_and_save(args):
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    print(f"--- 1. Чтение датасета ({args.nrows} строк max) ---")
    df = pd.read_csv(args.data_path, nrows=args.nrows).dropna(subset=["text", "topic"])

    print(f"--- 2. Балансировка по классам (до {args.samples_per_class} на класс) ---")
    df_balanced = sample_topics(df, topics, args.samples_per_class, args.random_state)
    print(f"   > Всего для обучения: {len(df_balanced)} текстов")

    print("--- 3. Чистка и лемматизация текстов ---")
    df_balanced["clean_text"] = df_balanced["text"].apply(clean_text)
    df_balanced = df_balanced[df_balanced["clean_text"].str.len() > 0]

    X_all = df_balanced["clean_text"].values
    y_all = df_balanced["topic"].values

    print("--- 4. Разделение на Train/Test ---")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_all,
    )
    print(f"   > Train: {len(X_train_text)} | Test: {len(X_test_text)}")

    print("--- 5. TF-IDF + отбор признаков ---")
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        stop_words=STOP_WORDS,
        sublinear_tf=True,
    )
    X_train_raw = tfidf.fit_transform(X_train_text)

    k_best = min(args.k_best, X_train_raw.shape[1])
    if k_best <= 0:
        raise ValueError("k_best <= 0. Увеличьте nrows/samples_per_class или уменьшите min_df.")

    selector = SelectKBest(chi2, k=k_best)
    X_train_sel = selector.fit_transform(X_train_raw, y_train).toarray()
    feats = np.array(tfidf.get_feature_names_out())[selector.get_support()]
    print(f"   > Отобрано признаков: {len(feats)}")

    print("--- 6. Обучение FuzzyDecisionTree (tuned defaults) ---")
    start_time = time.time()
    model = FuzzyDecisionTree(max_depth=args.max_depth, min_gain=args.min_gain)
    model.fit(X_train_sel, y_train, feats)
    print(f"   > Обучение завершено за {time.time() - start_time:.1f} сек.")

    print("--- 7. Оценка на test ---")
    X_test_raw = tfidf.transform(X_test_text)
    X_test_sel = selector.transform(X_test_raw).toarray()
    y_pred = [model.predict(vec) for vec in X_test_sel]

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Отчет по классам:")
    print(classification_report(y_test, y_pred, zero_division=0))

    save_pipeline(args.model_path, model, tfidf, selector, topics)
    print(f"Модель сохранена в: {args.model_path}")


def predict_mode(args):
    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {args.model_path}\n"
            "Сначала запустите обучение: python3 solv_14.py --mode train"
        )
    pipeline = load_pipeline(args.model_path)

    if args.text:
        pred = predict_for_text(args.text, pipeline)
        print(f"Текст: {args.text[:120]}{'...' if len(args.text) > 120 else ''}")
        print(f"Класс: {pred}")
        return

    print("Введите текст для классификации:")
    raw_text = input("> ").strip()
    pred = predict_for_text(raw_text, pipeline)
    print(f"Класс: {pred}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuzzy-классификатор с более мягкими анти-overfit фиксами."
    )
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--nrows", type=int, default=250000)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--topics",
        type=str,
        default=",".join(DEFAULT_TOPICS),
        help="Список классов через запятую",
    )
    parser.add_argument("--samples-per-class", type=int, default=1800)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--k-best", type=int, default=4000)
    parser.add_argument("--max-depth", type=int, default=22)
    parser.add_argument("--min-gain", type=float, default=0.0015)
    parser.add_argument("--text", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train_and_save(args)
    else:
        predict_mode(args)
