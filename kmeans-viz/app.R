# ---- libraries ----
library(shiny)
library(tidyverse)
library(tidymodels)
library(cluster)

# ---- data loading ----
# Files should live in ./dataset/ relative to app.R
df <- readr::read_csv("dataset/df.csv")
df$distance_miles <- round(df$distance_miles, digits = 3)

# ---- data preprocessing ----
cluster_df <- df %>%
  select(Road_traffic_density, distance_miles, City, Time_Orderd, `Time_taken (min)`) %>%
  mutate(
    city_encode = case_when(
      City == "Semi-Urban" ~ 1,
      City == "Urban" ~ 2,
      TRUE ~ 3
    ),
    Road_traffic_density = case_when(
      Road_traffic_density == "High" ~ "Jam",
      TRUE ~ Road_traffic_density
    ),
    traffic_encode = case_when(
      Road_traffic_density == "Low" ~ 1,
      Road_traffic_density == "High" ~ 3,
      TRUE ~ 4
    )
  )

cluster_df$day_hour <- as.numeric(lubridate::hour(cluster_df$Time_Orderd))
cluster_df$Road_traffic_density <- factor(
  cluster_df$Road_traffic_density,
  levels = c("Jam", "High", "Medium", "Low")
)
cluster_df$City <- factor(
  cluster_df$City,
  levels = c("Metropolitian", "Urban", "Semi-Urban")
)

# ---- silhouette table (precomputed) ----
K_score <- readr::read_csv("dataset/K_Silhouette.csv")

# ---- helper objects & functions ----
options1 <- c("Road_traffic_density", "City")
options2 <- c("Time of Day", "Delivery Distance (miles)")
color_palette <- c("red", "steelblue", "lightgreen")

cluster_f <- function(cluster_df, k) {
  x <- cluster_df %>%
    select(-c(Time_Orderd, City, Road_traffic_density))
  
  kmeans(x, centers = k) %>%
    augment(cluster_df)
}

final_df <- function(df, k) {
  df <- cluster_f(df, k)
  df %>%
    left_join(
      df %>%
        group_by(.cluster) %>%
        summarise(avg_time = mean(`Time_taken (min)`), .groups = "drop"),
      by = ".cluster"
    )
}

scatter <- function(df, x, color_type) {
  ggplot(
    df,
    aes(
      x = x,
      y = `Time_taken (min)`,
      group = .cluster
    )
  ) +
    geom_point(aes(color = color_type)) +
    scale_color_manual(values = color_palette) +
    facet_wrap(~reorder(.cluster, avg_time)) +
    labs(
      color = "Traffic / City",
      x = "",
      y = "Time taken (min)"
    )
}

# ---- UI ----
ui <- fluidPage(
  titlePanel("K-Means Clustering of Zomato Delivery Patterns"),
  h3("Interactive visualization of how traffic, distance, city type, and time-of-day influence delivery time"),
  br(),
  div(
    style = "font-size: small; color: #555555;",
    p("Zomato is a major Indian app-based food delivery platform where delivery time is shaped by factors such as traffic congestion, distance traveled, and the time of day an order is placed. This Shiny app applies K-means clustering to a subset of Zomato’s delivery data to uncover hidden structure in these patterns and identify which conditions most strongly influence delivery duration. The purpose of the visualization is to make the clustering process interpretable while also helping customers optimize when they order food by revealing peak-hour delays and the impact of traffic on delivery speeds.")
  ),
  br(),
  div(
    style = "font-size: small; color: #555555;",
    p("Users can adjust the number of clusters (K), switch between x-axis variables such as time-of-day or distance, and color deliveries by traffic level or city type to explore how different operational factors shape delivery behavior. Faceted scatterplots display each cluster separately, and the silhouette score provides immediate feedback on how well the clusters separate. Together, these interactive elements allow users to investigate meaningful behavioral patterns—such as lunch and dinner rush peaks, the dominant effect of congestion on delivery time, and the weaker influence of distance or city type—offering both model interpretability and actionable insight for smarter food-ordering decisions.")
  ),
  br(),
  sliderInput("k", "Select number of K", min = 2, max = 10, value = 2),
  selectInput("x", "x-axis", choices = options2),
  selectInput("traffic_city", "Choose color variable", choices = options1),
  br(),
  plotOutput("scatter"),
  br(),
  tableOutput("kscore")
)

# ---- server ----
server <- function(input, output, session) {
  
  df_reactive <- reactive({
    req(input$k)
    final_df(cluster_df, input$k)
  })
  
  x_axis <- reactive({
    req(input$x)
    if (input$x == "Time of Day") {
      df_reactive()$day_hour
    } else {
      df_reactive()$distance_miles
    }
  })
  
  k_table <- reactive({
    req(input$k)
    filter(K_score, K == input$k)
  })
  
  color_filter <- reactive({
    req(input$traffic_city)
    if (input$traffic_city == "Road_traffic_density") {
      df_reactive()$Road_traffic_density
    } else {
      df_reactive()$City
    }
  })
  
  output$scatter <- renderPlot({
    scatter(
      df = df_reactive(),
      x = x_axis(),
      color_type = color_filter()
    )
  })
  
  output$kscore <- renderTable({
    k_table()
  })
}

# ---- run app ----
shinyApp(ui = ui, server = server)
