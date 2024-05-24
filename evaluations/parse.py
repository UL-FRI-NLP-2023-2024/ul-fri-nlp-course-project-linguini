doc2query = open("doc2query.txt", "r")
slot5 = open("slot5.txt", "r")

doc2querylines = doc2query.readlines()
slot5lines = slot5.readlines()

doc2query.close()
slot5.close()

count_beam_doc2query = {}
count_sampled_doc2query = {}
count_comb_doc2query = {}

count_beam_t5 = {}
count_sampled_t5 = {}
count_comb_t5 = {}


for line in doc2querylines:
	parts = line.split()
	beam_hits = int(parts[0].split("/")[0])
	sampled_hits = int(parts[1].split("/")[0])
	total_hits = beam_hits+sampled_hits
	#print(total_hits)

	if beam_hits not in count_beam_doc2query.keys():
		count_beam_doc2query[beam_hits] = 0
	count_beam_doc2query[beam_hits] += 1

	if sampled_hits not in count_sampled_doc2query.keys():
		count_sampled_doc2query[sampled_hits] = 0
	count_sampled_doc2query[sampled_hits] += 1

	if total_hits not in count_comb_doc2query.keys():
		count_comb_doc2query[total_hits] = 0
	count_comb_doc2query[total_hits] += 1

for line in slot5lines:
	parts = line.split()
	beam_hits = int(parts[0].split("/")[0])
	sampled_hits = int(parts[1].split("/")[0])
	total_hits = beam_hits+sampled_hits
	#print(total_hits)

	if beam_hits not in count_beam_t5.keys():
		count_beam_t5[beam_hits] = 0
	count_beam_t5[beam_hits] += 1

	if sampled_hits not in count_sampled_t5.keys():
		count_sampled_t5[sampled_hits] = 0
	count_sampled_t5[sampled_hits] += 1

	if total_hits not in count_comb_t5.keys():
		count_comb_t5[total_hits] = 0
	count_comb_t5[total_hits] += 1

#normalise counts
def normalise_counts(dictionary):
	sumall = sum(dictionary.values())
	for key in dictionary.keys():
		dictionary[key] /= sumall

from matplotlib import pyplot as plt

count_sampled_doc2query[5] = 0
count_comb_doc2query[8] = 0
count_comb_doc2query[9] = 0
count_comb_doc2query[10] = 0

normalise_counts(count_beam_doc2query)
plt.bar(count_beam_doc2query.keys(), count_beam_doc2query.values())
plt.title("Proportion of apropriate questions out of 5 for beam search doc2query")
plt.xlabel("number of appropriate questions out of 5")
plt.show()

normalise_counts(count_sampled_doc2query)
plt.bar(count_sampled_doc2query.keys(), count_sampled_doc2query.values())
plt.title("Proportion of apropriate questions out of 5 for sampled doc2query")
plt.xlabel("number of appropriate questions out of 5")
plt.show()

normalise_counts(count_comb_doc2query)
plt.bar(count_comb_doc2query.keys(), count_comb_doc2query.values())
plt.title("Proportion of apropriate questions out of 5 for combined sampled and beam search doc2query")
plt.xlabel("number of appropriate questions out of 5")
plt.show()

#-----------------------------------------------------------

normalise_counts(count_beam_t5)
plt.bar(count_beam_t5.keys(), count_beam_t5.values())
plt.title("Proportion of apropriate questions out of 5 for beam search slot5")
plt.xlabel("number of appropriate questions out of 5")
plt.show()

normalise_counts(count_sampled_t5)
plt.bar(count_sampled_t5.keys(), count_sampled_t5.values())
plt.title("Proportion of apropriate questions out of 5 for sampled slot5")
plt.xlabel("number of appropriate questions out of 5")
plt.show()

normalise_counts(count_comb_t5)
plt.bar(count_comb_t5.keys(), count_comb_t5.values())
plt.title("Proportion of apropriate questions out of 5 for combined sampled and beam search slot5")
plt.xlabel("number of appropriate questions out of 5")
plt.show()


def dict_avg(dictionary):
	num_vals = sum(dictionary.values())
	sumscores = 0
	for key in dictionary.keys():
		sumscores += key*dictionary[key]
	return sumscores/num_vals

print(f"Average score for beam search doc2query {dict_avg(count_beam_doc2query)}")
print(f"Average score for sampled doc2query {dict_avg(count_sampled_doc2query)}")
print(f"Average score for combined doc2query {dict_avg(count_comb_doc2query)}")
print()
print(f"Average score for beam search slot5 {dict_avg(count_beam_t5)}")
print(f"Average score for sampled slot5 {dict_avg(count_sampled_t5)}")
print(f"Average score for combined slot5 {dict_avg(count_comb_t5)}")