from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import traceback

# Set up WebDriver
driver = webdriver.Firefox()

url = "https://www.rottentomatoes.com/browse/movies_at_home/affiliates:hulu"

try:
    driver.get(url)
    time.sleep(2)  # Allow time for the page to load

    movies = []
    max_movies = 30  # Max number of movies to scrape
    load_more_selector = 'button[data-qa="dlp-load-more-button"]'  # Selector for the "Load More" button
    movie_selector = 'div[data-qa="discovery-media-list-item"]'  # Selector for movie containers
    title_selector = 'span[data-qa="discovery-media-list-item-title"]'  # Selector for movie titles
    tomatometer_selector = 'rt-text[slot="criticsScore"]'  # Selector for the Tomatometer
    popcornmeter_selector = 'rt-text[slot="audienceScore"]'  # Selector for the Popcornmeter
    streaming_start_selector = 'span.smaller[data-qa="discovery-media-list-item-start-date"]'  # Selector for the Streaming Start Date

    while len(movies) < max_movies:
        # Find movie elements on the page
        try:
            movie_elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, movie_selector))
            )
        except Exception as e:
            print(f"Error locating movie elements: {e}")
            break

        for movie_element in movie_elements:
            try:
                # Extract movie title
                try:
                    title_element = movie_element.find_element(By.CSS_SELECTOR, title_selector)
                    movie_title = title_element.text.strip()
                except Exception as e:
                    movie_title = "N/A"
                    print(f"Error extracting movie title: {e}")

                # Extract Tomatometer score
                try:
                    tomatometer_element = movie_element.find_element(By.CSS_SELECTOR, tomatometer_selector)
                    tomatometer_score = tomatometer_element.text.strip()
                except Exception as e:
                    tomatometer_score = "N/A"
                    print(f"Error extracting Tomatometer score: {e}")

                # Extract Popcornmeter score
                try:
                    popcornmeter_element = movie_element.find_element(By.CSS_SELECTOR, popcornmeter_selector)
                    popcornmeter_score = popcornmeter_element.text.strip()
                except Exception as e:
                    popcornmeter_score = "N/A"
                    print(f"Error extracting Popcornmeter score: {e}")

                # Extract Streaming Start Date
                try:
                    streaming_start_element = movie_element.find_element(By.CSS_SELECTOR, streaming_start_selector)
                    streaming_start_date = streaming_start_element.text.strip()
                except Exception as e:
                    streaming_start_date = "N/A"
                    print(f"Error extracting Streaming Start Date: {e}")

                # Add movie data if the title is not already recorded
                if movie_title and not any(movie['Title'] == movie_title for movie in movies):
                    movies.append({
                        "Title": movie_title,
                        "Tomatometer": tomatometer_score,
                        "Popcornmeter": popcornmeter_score,
                        "Streaming Start": streaming_start_date
                    })

                # Break if the maximum movie count is reached
                if len(movies) >= max_movies:
                    break

            except Exception as e:
                print(f"Error processing a movie element: {e}")

        # Click the "Load More" button if available
        try:
            load_more_button = driver.find_element(By.CSS_SELECTOR, load_more_selector)
            driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
            ActionChains(driver).move_to_element(load_more_button).click(load_more_button).perform()
            time.sleep(2)  # Wait for new movies to load
        except Exception as e:
            print("No more 'Load More' button or an error occurred:", e)
            break  # Exit the loop if no more button is available


    if movies:
        # Create a DataFrame
        df = pd.DataFrame(movies)

        # Clean data
        df["Tomatometer"] = df["Tomatometer"].str.replace("%", "").replace("N/A", None).astype("float", errors="ignore")
        df["Popcornmeter"] = df["Popcornmeter"].str.replace("%", "").replace("N/A", None).astype("float",
                                                                                                 errors="ignore")
        df["Streaming Start"] = df["Streaming Start"].str.replace("Streaming ", "")
        df["Streaming Start"] = pd.to_datetime(df["Streaming Start"], format="%b %d, %Y", errors="coerce")
        df['Streaming Start'] = df['Streaming Start'].dt.strftime('%d.%m.%Y') #change format
        #handle missing cols by mean of the columns
        cols = ['Tomatometer', 'Popcornmeter']
        df[cols] = df[cols].apply(lambda x: x.fillna(x.mean()), axis=0)

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Save to a CSV file
        df.to_csv("hulu_movies.csv")
        print(f"Movies saved to 'hulu_movies.csv' with {len(df)} unique movies.")
    else:
        print("No movies were scraped.")



except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()

finally:
    driver.quit()
