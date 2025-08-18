import json
import matplotlib.pyplot as plt
import seaborn as sns


def count_words(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    words = 0
    for seg in data["segments"]:
        words += len(seg["text"].split())

    return words

def plot_bar(transcript_paths, wrong_words):
    transcripts = [path.split("/")[-1] for path in transcript_paths]
    accuracies = [(1 - wrong / count_words(path)) * 100 for path, wrong in zip(transcript_paths, wrong_words)]

    plt.figure(figsize=(12, 6))
    plt.bar(transcripts, accuracies, color="skyblue")
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy (%)")
    plt.title("Transcript Accuracy")
    plt.tight_layout()
    plt.savefig("bar_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_box(transcript_paths, wrong_words):
    accuracies = [(1 - wrong / count_words(path)) * 100 for path, wrong in zip(transcript_paths, wrong_words)]
    sns.boxplot(data=accuracies, orient="h")
    plt.xlabel("Accuracy (%)")
    plt.title("Distribution of Transcript Accuracy")
    plt.savefig("box_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_histogram(transcript_paths, wrong_words):
    accuracies = [(1 - wrong / count_words(path)) * 100 for path, wrong in zip(transcript_paths, wrong_words)]
    plt.hist(accuracies, bins=10, color="cornflowerblue", edgecolor="black")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Number of Transcripts")
    plt.title("Histogram of Transcript Accuracies")
    plt.savefig("histogram_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_scatter(transcript_paths, wrong_words):
    total_words_list = [count_words(path) for path in transcript_paths]
    error_rates = [wrong / total for wrong, total in zip(wrong_words, total_words_list)]

    plt.scatter(total_words_list, error_rates, alpha=0.7)
    plt.xlabel("Total Words")
    plt.ylabel("Error Rate")
    plt.title("Error Rate vs Transcript Length")
    plt.grid(True)
    plt.savefig("scatter_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    TRANSCRIPT_PATHS = [
        "../Transcriptions/ST4/TPP22/PTP0830/2019-04-04_PTP0830_6.Sitzung.json",
        "../Transcriptions/ST4/TPP10/PTP1342/2019-07-01_PTP1342_10. Sitzung.json",
        "../Transcriptions/ST4/TPP48/PT0125/2018-07-26_PT0125_2. Sitzung.json",
        "../Transcriptions/ST4/TPP18/PTP0764 (PT0018)/2017-11-14_PTP0764_7. Sitzung Teil 2.json",
        "../Transcriptions/ST4/TPP18/PTP1678 (PT0079)/2018-05-29_PTP1678_13. Sitzung Teil 1.json",
        "../Transcriptions/ST4/TPP15/PTP0425/2020-05-08_PTP0425_12. Sitzung_Teil 1.json",
        "../Transcriptions/ST4/TPP15/PTP0167/2020-10-27_PTP0167_7. Sitzung_Teil 2.json",
        "../Transcriptions/ST4/TPP15/PTP1496/2020-07-07_PTP1496_9. Sitzung Teil 1.json"
    ]

    wrong_words = [224, 165, 73, 14, 197, 290, 77, 481]

    with open("evaluation.txt", "w", encoding="utf-8") as f:
        for i in range(len(TRANSCRIPT_PATHS)):
            total_words = count_words(TRANSCRIPT_PATHS[i])

            f.write(f"{TRANSCRIPT_PATHS[i]}: {wrong_words[i]} out of {total_words} words wrong\n"
                    f"Accuracy: {(1 - wrong_words[i]/total_words) * 100:.2f}%\n")

    plot_bar(TRANSCRIPT_PATHS, wrong_words)
    plot_box(TRANSCRIPT_PATHS, wrong_words)
    plot_histogram(TRANSCRIPT_PATHS, wrong_words)
    plot_scatter(TRANSCRIPT_PATHS, wrong_words)


