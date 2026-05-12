# %%
class CurrencyAccount:
    def __init__(self, currency: str, balance: float) -> None:
        if not currency:
            raise ValueError("Currency cannot be empty")

        if balance < 0:
            raise ValueError("Balance cannot be negative")

        self.__currency: str = currency
        self.__balance: float = balance

    @property
    def balance(self) -> float:
        return self.__balance
    
    @property
    def currency(self) -> str:
        return self.__currency
    
    def __str__(self) -> str:
        return f"{self.balance:.2f} {self.currency}"
    
    def check_currency(self, currency: str) -> bool:
        if self.currency != currency:
            print("Currency doesn't support")
            return False

        return True

    def deposit(self, amount: float, currency: str) -> None:
        if amount <= 0:
            print("Amount must be positive")
            return None

        if not self.check_currency(currency):
            return None

        self.__balance = self.balance + amount

    def withdraw(self, amount: float, currency: str) -> None:
        if amount <= 0:
            print("Amount must be positive")
            return None

        if not self.check_currency(currency):
            return None

        if self.balance >= amount:
            self.__balance = self.balance - amount
        else:
            print("Not enough money")
            return None


# BankAccount
class BankAccount:
    EXCHANGE_RATES = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.08,
        ("USD", "RUB"): 90,
        ("EUR", "RUB"): 98,
    }
    
    def __init__(self, owner: str) -> None:
        self.__owner: str = owner
        self.__accounts: dict[str, CurrencyAccount] = {}
        self.history: list[str] = []

    def __str__(self) -> str:
        accounts_info = []

        for account in self.__accounts.values():
            accounts_info.append(str(account))

        return f"{self.owner}: " + ", ".join(accounts_info)
        
    @property
    def owner(self) -> str:
        return self.__owner
    
    def add_currency_account(self, currency: str, balance: float) -> None:
        if currency in self.__accounts:
            print("Account in this currency already exists")
            return None

        self.__accounts[currency] = CurrencyAccount(currency, balance)

    def check_currency(self, currency: str) -> bool:
        if currency not in self.__accounts:
            print(f"{self.owner} hasn`t got account in {currency}")
            return False

        return True
    
    def print_history(self) -> None:
        if not self.history:
            print("History is empty")
            return None

        print(f"{self.owner} history:")

        for operation in self.history:
            print(f"- {operation}")

    def get_account_by_currency(self, currency: str) -> CurrencyAccount | None:
        if not self.check_currency(currency):
            return None
        
        return self.__accounts[currency]
    
    def transfer_to_same_currency_account(self, other: "BankAccount", amount: float, currency: str) -> None:
        if amount <= 0:
            print("Amount must be positive")
            return None
    
        if not self.check_currency(currency) or not other.check_currency(currency):
            return None
        
        sender_account = self.get_account_by_currency(currency)
        recipient_account = other.get_account_by_currency(currency)

        if sender_account.balance < amount:
            print("Not enough money")
            return None

        sender_account.withdraw(amount, currency)
        recipient_account.deposit(amount, currency)

        print(f'{self.owner} sent {amount} {currency} to {other.owner}')
        self.history.append(f'Transfer to {other.owner}: {amount} {currency}')
        other.history.append(f'Deposit from {self.owner}: {amount} {currency}')
    
    def get_exchange_rate(self, sender_currency: str, recipient_currency: str) -> float:
        direct_key = (sender_currency, recipient_currency)
        reverse_key = (recipient_currency, sender_currency)

        if direct_key in self.EXCHANGE_RATES:
            return self.EXCHANGE_RATES[direct_key]

        if reverse_key in self.EXCHANGE_RATES:
            return 1 / self.EXCHANGE_RATES[reverse_key]

        raise ValueError("Exchange rate doesn't exist")

    def transfer_to_with_conversion(self, other: "BankAccount", amount: float, sender_currency: str, recipient_currency: str) -> None:

        if (sender_currency == recipient_currency):
            self.transfer_to_same_currency_account(other, amount, sender_currency)
            return None
        
        if amount <= 0:
            print("Amount must be positive")
            return None
        
        if not self.check_currency(sender_currency) or not other.check_currency(recipient_currency):
            return None

        rate = self.get_exchange_rate(sender_currency, recipient_currency)

        sender_account = self.get_account_by_currency(sender_currency)
        recipient_account = other.get_account_by_currency(recipient_currency)

        if sender_account.balance < amount:
            print("Not enough money")
            return None

        converted_sum: float = round(amount * rate, 2)

        sender_account.withdraw(amount, sender_currency)
        recipient_account.deposit(converted_sum, recipient_currency)

        print(f'{self.owner} sent {amount} {sender_currency} to {other.owner}, {other.owner} received {converted_sum} {recipient_currency}')
        self.history.append(f'Transfer to {other.owner}: {amount} {sender_currency} -> {converted_sum} {recipient_currency}')
        other.history.append(f'Deposit from {self.owner}: {converted_sum} {recipient_currency}')

    def get_total_amount(self, currency):
        total: float = 0

        for curr, account in self.__accounts.items():
            if curr == currency:
                total += account.balance
            else:
                rate = self.get_exchange_rate(curr, currency)
                total += account.balance * rate

        return f'{self.owner} has total sum in {currency}: {round(total, 2)}'
    

# %%
user1 = BankAccount("Anna")
user1.add_currency_account("USD", 100.67)
user1.add_currency_account("EUR", 50.78)

user2 = BankAccount("Valera")
user2.add_currency_account("USD", 50.50)
user2.add_currency_account("RUB", 5000.80)

print(user1)
print(user2)

user1.transfer_to_same_currency_account(user2, 10, "USD")
user1.transfer_to_same_currency_account(user2, 10, "RUB")

user2.transfer_to_same_currency_account(user1, 15, "EUR")
user2.transfer_to_same_currency_account(user1, 30, "USD")

user2.transfer_to_with_conversion(user1, 1000, "RUB", "USD")
user2.transfer_to_with_conversion(user1, 1000, "RUB", "EUR")

user1.transfer_to_with_conversion(user1, 1000, "USD", "RUB")

print(user1)
print(user2)

user1.print_history()

user1.get_account_by_currency("USD").balance
user1.get_total_amount("RUB")




# %%
