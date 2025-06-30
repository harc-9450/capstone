import pandas as pd

def generate_movie_summaries(full_data):

    summaries = []
    movie_groups = full_data.groupby('MovieID')

    for movie_id, group in movie_groups:
        title = group['Title'].iloc[0]
        genre = group['Genres'].iloc[0]
        avg_rating = round(group['Rating'].mean(), 2)
        rating_count = group['Rating'].count()

        # Most frequent user demographics
        top_gender = group['Gender'].mode()[0]
        top_age = group['Age'].mode()[0]
        top_occupation = group['Occupation'].mode()[0]

        # Generate summary
        summary = (
            f"{title} is a {genre} movie rated an average of {avg_rating} "
            f"by {rating_count} users. Most common viewers are {top_gender}, "
            f"aged {top_age}, with occupation ID {top_occupation}."
        )

        summaries.append({
            "MovieID": movie_id,
            "Title": title,
            "Summary": summary
        })

    summary_df = pd.DataFrame(summaries)
    print(f"✅ Generated summaries for {len(summary_df)} movies.")
    return summary_df