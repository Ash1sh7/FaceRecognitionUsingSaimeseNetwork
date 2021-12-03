from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

# Define any person(celb) you want :)
persons = ["Harrison Ford",
           "Ed Westwick",
           "Chris Hemsworth",
           "Nicole Kidman",
           "Shah Rukh Khan",
           "Roger Moore",
           "Jason Statham",
           "Megan Fox",
           "Marisa Tomei",
           "Nicolas Cage",
           "Britney Spears",
           "Reese Witherspoon"]

person_keywords = ""
for person in persons:
    person_keywords += person + ","

# creating list of arguments
arguments = {"keywords":person_keywords,
             "limit":36,
             "print_urls":True}

# passing the arguments to the function
paths = response().download(keywords=person_keywords,limit=36)