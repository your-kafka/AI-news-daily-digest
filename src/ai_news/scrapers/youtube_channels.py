from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Channel:
    name: str
    channel_id: str
    ai_focused: bool = True


CHANNELS: tuple[Channel, ...] = (
    Channel("Two Minute Papers", "UCbfYPyITQ-7l4upoX8nvctg"),
    Channel("AI Explained", "UCNJ1Ymd5yFuUPtn21xtRbbw"),
    Channel("Andrej Karpathy", "UCXUPKJO5MZQN11PqgIvyuvQ"),
    Channel("Yannic Kilcher", "UCZHmQk67mSJgfCCTn7xBfew"),
    Channel("Machine Learning Street Talk", "UCMLtBahI5DMrt0NPvDSoIRQ"),
    Channel("Anthropic", "UCrDwWp7EBBv4NwvScIpBDOA"),
    Channel("OpenAI", "UCXZCJLdBC09xxGZ6gcdrc6A"),
    Channel("Google DeepMind", "UCP7jMXSY2xbc3KCAE0MHQ-A"),
    Channel("Hugging Face", "UCHlNU7kIZhRgSbhHvFoy72w"),
    Channel("Sam Witteveen", "UC55ODQSvARtgSyc8ThfiepQ"),
    Channel("bycloud", "UCgfe2ooZD3VJPB6aJAnuQng"),
    Channel("Matthew Berman", "UCawZsQWqfGSbCI5yjkdVkTA"),
    Channel("Latent Space", "UCxBcwypKK-W3GHd_RZ9FZrQ"),
    Channel("AI Coffee Break with Letitia", "UCobqgqE4i5Kf7wrxRxhToQA"),
    Channel("DeepLearningAI", "UCcIXc5mJsHVYTZR1maL5l9w"),
    Channel("No Priors", "UCSI7h9hydQ40K5MJHnCrQvw"),
    Channel("Cognitive Revolution", "UCjNRVMBVI30Sak_p6HRWhIA"),
    Channel("The AI Daily Brief", "UCKelCK4ZaO6HeEI1KQjqzWA"),
    Channel("Wes Roth", "UCqcbQf6yw5KzRoDDcZ_wBSw"),
    Channel("Matt Wolfe", "UChpleBmo18P08aKCIgti38g"),
    Channel("TheAIGRID", "UCbY9xX3_jW5c2fjlZVBI4cg"),
    Channel("Nate B Jones", "UC0C-17n9iuUQPylguM1d-lQ"),
    Channel("Prompt Engineering", "UCDq7SjbgRKty5TgGafW8Clg"),
    Channel("Umar Jamil", "UCtAcpQcYerN8xxZJYTfWBMw"),
    Channel("Cole Medin", "UCMwVTLZIRRUyyVrkjDpn4pA"),
    Channel("Dwarkesh Patel", "UCXl4i9dYBrFOabk0xGmbkRA", ai_focused=False),
    Channel("Lex Fridman", "UCSHZKyawb77ixDdsGog4iWA", ai_focused=False),
    Channel("Y Combinator", "UCcefcZRL2oaA_uBNeo5UOWg", ai_focused=False),
    Channel("Fireship", "UCsBjURrPoezykLs9EqgamOA", ai_focused=False),
    Channel("IBM Technology", "UCKWaEZ-_VweaEx1j62do_vQ", ai_focused=False),
    Channel("3Blue1Brown", "UCYO_jab_esuFRV4b17AJtAw", ai_focused=False),
)
