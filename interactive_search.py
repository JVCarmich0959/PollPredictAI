import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


def load_data(path: str = "Voter_Data.csv") -> pd.DataFrame:
    """Load voter data from a CSV file."""
    return pd.read_csv(path)


def filter_data(df: pd.DataFrame, race: str = None, party: str = None,
                gender: str = None, place: str = None) -> pd.DataFrame:
    """Return subset of DataFrame matching the provided criteria."""
    query_parts = []
    if race:
        query_parts.append(f"`race_code` == '{race}'")
    if party:
        query_parts.append(f"`party_cd` == '{party}'")
    if gender:
        query_parts.append(f"`gender_code` == '{gender}'")
    if place:
        query_parts.append(f"`polling_place` == '{place}'")

    if query_parts:
        return df.query(" & ".join(query_parts))
    return df


def show_summary(df: pd.DataFrame) -> None:
    """Display summary information and a party distribution plot."""
    if df.empty:
        print("No results found")
        return

    display(df.head())
    display(df.describe(include='all'))

    counts = df['party_cd'].value_counts()
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Party Distribution in Results')
    plt.xlabel('Party')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def create_search_widget(df: pd.DataFrame) -> widgets.VBox:
    """Build and return the interactive search widget."""
    race_w = widgets.Text(description='Race:')
    party_w = widgets.Text(description='Party:')
    gender_w = widgets.Text(description='Gender:')
    place_w = widgets.Text(description='Place:')
    search_btn = widgets.Button(description='Search')
    output = widgets.Output()

    def on_click(_):
        with output:
            clear_output()
            filtered = filter_data(df, race_w.value or None, party_w.value or None,
                                   gender_w.value or None, place_w.value or None)
            show_summary(filtered)

    search_btn.on_click(on_click)
    box = widgets.VBox([race_w, party_w, gender_w, place_w, search_btn, output])
    return box


def main() -> None:
    df = load_data()
    widget = create_search_widget(df)
    display(widget)


if __name__ == '__main__':
    main()
