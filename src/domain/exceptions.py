"""Domain Exceptions.
Independent of HTTP status codes or framework-specific errors.
"""

class DomainError(Exception):
    """Base exception for all domain errors."""
    def __init__(self, message: str = "A domain error occurred."):
        self.message = message
        super().__init__(self.message)


DomainException = DomainError


class EntityNotFoundError(DomainError):
    """Raised when an entity is not found in repositories."""
    def __init__(self, entity_name_or_message: str, entity_id: str | int | None = None):
        if entity_id is not None:
            self.entity_name = entity_name_or_message
            self.entity_id = entity_id
            super().__init__(f"{entity_name_or_message} with identifier '{entity_id}' not found.")
        else:
            self.entity_name = "Entity"
            self.entity_id = ""
            super().__init__(entity_name_or_message)


class EntityAlreadyExistsError(DomainError):
    """Raised when an entity with same identifier/email already exists."""
    def __init__(self, entity_name: str, key: str, value: str):
        self.entity_name = entity_name
        self.key = key
        self.value = value
        super().__init__(f"{entity_name} with {key}='{value}' already exists.")


class InvalidCredentialsError(DomainError):
    """Raised when authentication credentials are invalid."""
    def __init__(self, message: str = "Invalid email or password."):
        super().__init__(message)


class GuardrailViolationError(DomainError):
    """Raised when user query violates security or academic guardrails."""
    def __init__(self, reasoning: str, feedback: str):
        self.reasoning = reasoning
        self.feedback = feedback
        super().__init__(f"Guardrail violation: {reasoning}")


class LLMProviderError(DomainError):
    """Raised when LLM provider fails or is unreachable."""
    def __init__(self, provider: str, details: str):
        self.provider = provider
        self.details = details
        super().__init__(f"LLM Provider '{provider}' error: {details}")
