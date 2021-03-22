def convertline(text, max_length=20):
    text=text.split("\t") #split the conversation by tabs
    inputs, targets=[],[] #create empty arrays for inputs and targets
    for y in range(1,len(text)): #iterate through the split conversation
        x=y-max_length if y-max_length >= 0 else 0 #get the starting value; if it's negative, use 0 instead
        inputs.append("/b".join(text[x:y])) #append to the inputs the current window, joined by /b
        targets.append(text[y]) #append the target
    return [f"{inputs[i]}\t{targets[i]}" for i in range(len(inputs))] #zip them together in a dict of inputs and targets

with open("src/testfile.txt", "r") as f:
    line = f.readline()
    while line:
        print(convertline(line.strip()))
        line=f.readline()