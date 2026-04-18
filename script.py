def get_age_group(age):
    if age < 0:
        return None
    if age < 20:
        return "young"
    if age < 40:
        return "young adult"
    return "old"


def main():
    name = input("Enter your name: ").strip()

    try:
        age = int(input("Enter your age: ").strip())
    except ValueError:
        print("Please enter a valid whole number for age.")
        return

    age_group = get_age_group(age)
    if age_group is None:
        print("Age cannot be negative.")
        return

    print(f"Hi {name}, you are {age_group}.")


if __name__ == "__main__":
    main()

